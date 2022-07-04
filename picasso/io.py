"""
    picasso.io
    ~~~~~~~~~~

    General purpose library for handling input and output of files

    :author: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""
import os.path as _ospath
import numpy as _np
import yaml as _yaml
import glob as _glob
import h5py as _h5py
import re as _re
import struct as _struct
import json as _json
import os as _os
import threading as _threading
from PyQt5.QtWidgets import QMessageBox as _QMessageBox
from . import lib as _lib
import abc
import nd2
from nd2reader import ND2Reader
from nd2reader.label_map import LabelMap
from nd2reader.raw_metadata import RawMetadata
from nd2reader.common_raw_metadata import parse_roi_shape, parse_roi_type, parse_dimension_text_line


class NoMetadataFileError(FileNotFoundError):
    pass


def _user_settings_filename():
    home = _ospath.expanduser("~")
    return _ospath.join(home, ".picasso", "settings.yaml")


def load_raw(path, prompt_info=None):
    try:
        info = load_info(path)
    except FileNotFoundError as error:
        if prompt_info is None:
            raise error
        else:
            result = prompt_info()
            if result is None:
                return
            else:
                info, save = result
                info = [info]
                if save:
                    base, ext = _ospath.splitext(path)
                    info_path = base + ".yaml"
                    save_info(info_path, info)
    dtype = _np.dtype(info[0]["Data Type"])
    shape = (info[0]["Frames"], info[0]["Height"], info[0]["Width"])
    movie = _np.memmap(path, dtype, "r", shape=shape)
    if info[0]["Byte Order"] != "<":
        movie = movie.byteswap()
        info[0]["Byte Order"] = "<"
    return movie, info


def save_config(CONFIG):
    this_file = _ospath.abspath(__file__)
    this_directory = _ospath.dirname(this_file)
    with open(_ospath.join(this_directory, "config.yaml"), "w") as config_file:
        _yaml.dump(CONFIG, config_file, width=1000)


def save_raw(path, movie, info):
    movie.tofile(path)
    info_path = _ospath.splitext(path)[0] + ".yaml"
    save_info(info_path, info)


def multiple_filenames(path, index):
    base, ext = _ospath.splitext(path)
    filename = base + "_" + str(index) + ext
    return filename


def load_tif(path):
    movie = TiffMultiMap(path, memmap_frames=False)
    info = movie.info()
    return movie, [info]

def load_nd2(path):
    movie = ND2Movie(path)
    info = movie.info()
    return movie, [info]

def load_movie(path, prompt_info=None):
    base, ext = _ospath.splitext(path)
    ext = ext.lower()
    if ext == ".raw":
        return load_raw(path, prompt_info=prompt_info)
    elif ext == ".tif":
        return load_tif(path)
    elif ext == '.nd2':
        return load_nd2(path)


def load_info(path, qt_parent=None):
    path_base, path_extension = _ospath.splitext(path)
    filename = path_base + ".yaml"
    try:
        with open(filename, "r") as info_file:
            info = list(_yaml.load_all(info_file, Loader=_yaml.FullLoader))
    except FileNotFoundError as e:
        print(
            "\nAn error occured. Could not find metadata file:\n{}".format(
                filename
            )
        )
        if qt_parent is not None:
            _QMessageBox.critical(
                qt_parent,
                "An error occured",
                "Could not find metadata file:\n{}".format(filename),
            )
        raise NoMetadataFileError(e)
    return info


def load_user_settings():
    settings_filename = _user_settings_filename()
    settings = None
    try:
        settings_file = open(settings_filename, "r")
    except FileNotFoundError:
        return _lib.AutoDict()
    try:
        settings = _yaml.load(settings_file, Loader=_yaml.FullLoader)
        settings_file.close()
    except Exception as e:
        print(e)
        print("Error reading user settings, Reset.")
    if not settings:
        return _lib.AutoDict()
    return _lib.AutoDict(settings)


def save_info(path, info, default_flow_style=False):
    with open(path, "w") as file:
        _yaml.dump_all(info, file, default_flow_style=default_flow_style)


def _to_dict_walk(node):
    """ Converts mapping objects (subclassed from dict)
    to actual dict objects, including nested ones
    """
    node = dict(node)
    for key, val in node.items():
        if isinstance(val, dict):
            node[key] = _to_dict_walk(val)
    return node


def save_user_settings(settings):
    settings = _to_dict_walk(settings)
    settings_filename = _user_settings_filename()
    _os.makedirs(_ospath.dirname(settings_filename), exist_ok=True)
    with open(settings_filename, "w") as settings_file:
        _yaml.dump(dict(settings), settings_file, default_flow_style=False)


class AbstractPicassoMovie(abc.ABC):
    """An abstract class defining the minimal interfaces of a PicassoMovie
    used throughout picasso.
    """
    @abc.abstractmethod
    def __init__(self):
        self.use_dask = False

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abc.abstractmethod
    def info(self):
        pass

    @abc.abstractmethod
    def camera_parameters(self, config):
        """Get the camera specific parameters:
            * gain
            * quantum efficiency
            * wavelength
        These parameters depend on camera settings (as described in metadata)
        but the values themselves are given in the config.yaml file.
        Each filetype (nd2, ome-tiff, ..) has their own structure of metadata,
        which needs to be matched in the config.yaml description, as detailed
        in the specific child classes.

        Args:
            config : dict
                description of camera parameters (for all possible settings)
                comes from the config.yaml file
        Returns:
            parameters : dict of lists of str
                keys: gain, qe, wavelength, cam_index, camera
        """
        return {'gain': [1], 'qe': [1], 'wavelength': [0], 'cam_index': 0,
                'camera': 'None'}

    @abc.abstractmethod
    def __getitem__(self, it):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        return self.n_frames

    def close(self):
        pass

    @abc.abstractmethod
    def get_frame(self, index):
        pass

    @abc.abstractmethod
    def tofile(self, file_handle, byte_order=None):
        pass

    @property
    @abc.abstractmethod
    def dtype(self):
        return 'u16'


class ND2Movie(AbstractPicassoMovie):
    """Subclass of the AbstractPicassoMovie to implement reading Nikon nd2
    files.
    Two packages for reading nd2 files have been tested and are used here:
    * nd2.ND2File - makes all metadata accessible, and uses dask arrays
      which is good for multiprocessing. Nonetheless, converting all data to
      numpy arrays takes long (2000 frames/s for dask planes; 20 frames/s with
      compute() on dask planes to get numpy)
    * nd2reader.ND2Reader - image reading is a bit faster than TiffMultiMap
      (ca 800 frames/s), and the file is serealizable for multiprocessing.
      However, very limited metadata is available.

    This class implements a hybrid version which uses both packages:
    nd2 for metadata retrieval, and nd2reader for image data retrieval
    """
    def __init__(self, path, verbose=False):
        super().__init__()
        if verbose:
            print("Reading info from {}".format(path))
        self.path = _ospath.abspath(path)
        nd2file = nd2.ND2File(path)
        self.sizes = nd2file.sizes

        required_dims = ['T', 'Y', 'X']  # exactly these, not more
        for dim in required_dims:
            if dim not in nd2file.sizes.keys():
                raise KeyError(
                    'Required dimension {:s} not in file {:s}'.format(
                        dim, self.path))
        if nd2file.ndim != len(required_dims):
            raise KeyError(
                'File {:s} has dimensions {:s} '.format(
                    self.path, str(nd2file.sizes.keys())) +
                'but should have exactly {:s}.'.format(str(required_dims)))

        self.meta = self.get_metadata(nd2file)

        self.nd2data = ND2Reader(self.path)
        self._shape = [
            self.nd2data.metadata['num_frames'],
            self.nd2data.metadata['width'],
            self.nd2data.metadata['height'],
            ]

    def info(self):
        return self.meta

    def get_metadata(self, nd2file):
        """Brings the file metadata in a readable form, and preprocesses it
        for easier downstream use.

        Args:
            nd2file : nd2.ND2File
                the object holding the image incl metadata
        Returns:
            info : dict
                the metadata
        """
        info = {
            # "Byte Order": self._tif_byte_order,
            "File": self.path,
            "Height": nd2file.sizes['Y'],
            "Width": nd2file.sizes['X'],
            "Data Type": nd2file.dtype.name,
            "Frames": nd2file.sizes['T'],
        }
        info['Acquisition Comments'] = ''

        mm_info = self.metadata_to_dict(nd2file)
        camera_name = mm_info.get('description', {}).get(
                'Metadata', {}).get('Camera Name', 'None')
        info['Camera'] = camera_name

        # simulate micro manager camera data for loading config values
        # see picasso/gui/localize:680ff
        # put into camera config
        # 'Sensitivity Categories': ['PixelReadoutRate', 'ReadoutMode']
        # 'Sensitivity':
        #     '540 MHz':
        #         'Rolling Shutter at 16-bit': sensitivityvalue  # or sensival directly behind 540 MHz?
        # 'Channel Device':
        #     'Name': 'Filter'
        #     'Emission Wavelengths':
        #         '2 (560)': 560
        readout_rate = mm_info.get(
                'description', {}).get('Metadata', {}).get(
                'Camera Settings', {}).get('Readout Rate', 'None')
        readout_mode = mm_info.get(
                'description', {}).get('Metadata', {}).get(
                'Camera Settings', {}).get('Readout Mode', 'None')
        filter = mm_info.get(
                'description', {}).get('Metadata', {}).get(
                'Camera Settings', {}).get('Microscope Settings', {}).get(
                'Nikon Ti2, FilterChanger(Turret-Lo)', 'None')

        sensitivity_category = 'PixelReadoutRate'
        info["Micro-Manager Metadata"] = {
            camera_name+'-'+sensitivity_category: readout_rate,
            'Filter': filter,
            }
        info["Picasso Metadata"] = {
            'Camera': camera_name,
            'PixelReadoutRate': readout_rate,
            'ReadoutMode': readout_mode,
            'Filter': filter,
        }
        info["nd2 Metadata"] = mm_info

        return info

    def metadata_to_dict(self, nd2file):
        """Extracts all types of metadata in the file and returns it in a dict.

        Args:
            nd2file : nd2.ND2File
                the object holding the image incl metadata
        Returns:
            mmmeta : dict
                all metadata
        """
        mmmeta = {}

        text_info = nd2file.text_info
        mmmeta['capturing'] = self.nikontext_to_dict(text_info['capturing'])
        mmmeta['AcquisitionDate'] = text_info['date']
        mmmeta['description'] = self.nikontext_to_dict(text_info['description'])
        mmmeta['optics'] = self.nikontext_to_dict(text_info['optics'])

        mmmeta['custom_data'] = nd2file.custom_data
        mmmeta['attributes'] = nd2file.attributes._asdict()
        mmmeta['metadata'] = self.nd2metadata_to_dict(nd2file.metadata)

        return mmmeta

    @classmethod
    def nikontext_to_dict(cls, text):
        """Some kinds of Nikon metadata are described with text, using
        newlines and colons. This function restructures the text into
        a dict.

        Args:
            text : str
                nikon-style metadata description text
        Returns:
            out : dict
                restructured text
        """
        out = {}
        curr_keys = []
        for i, item in enumerate(text.split('\r\n')):
            itparts = item.split(':')
            itparts = [it.strip() for it in itparts if it.strip()!='']
            if len(itparts)==1:
                curr_keys.append(itparts[0])
                cls.set_nested_dict_entry(out, curr_keys, {})
            elif len(itparts)==2:
                cls.set_nested_dict_entry(
                    out, curr_keys+[itparts[0]], itparts[1])
            elif len(itparts)==3:
                curr_keys.append(itparts[0])
                cls.set_nested_dict_entry(out, curr_keys, {})
                cls.set_nested_dict_entry(
                    out, curr_keys+[itparts[1]], itparts[2])
            elif len(itparts) > 3:
                curr_keys.append(itparts[0])
                cls.set_nested_dict_entry(out, curr_keys, {})
                cls.set_nested_dict_entry(
                    out, curr_keys+[itparts[1]], item)
                # raise KeyError(
                #     'Cannot parse three or more colons between newlines: ' +
                #     item)
        return out

    @classmethod
    def nd2metadata_to_dict(cls, meta):
        """Restructure the 'metadata' field from the package nd2 into a dict
        for independent use.
        https://github.com/tlambert03/nd2/blob/main/src/nd2/structures.py

        Args:
            meta : nd2 metadata structure
                the 'metadata' part of nd2 metadata
        Returns:
            out : dict
                the content as a dict.
        """
        out = {}
        out['contents'] = meta.contents.__dict__
        chans = [{}] * len(meta.channels)
        for i, chan in enumerate(meta.channels):
            chans[i] = chan.__dict__
            metachan = chan.__dict__['channel'].__dict__
            chans[i]['channel'] = {}
            for k, v in metachan.items():
                chans[i]['channel'][str(k)] = str(v)
            chans[i]['loops'] = chan.__dict__['loops'].__dict__
            chans[i]['microscope'] = chan.__dict__['microscope'].__dict__
            chans[i]['volume'] = chan.__dict__['volume'].__dict__
            axints = chans[i]['volume']['axesInterpretation']
            chans[i]['volume']['axesInterpretation'] = [None]*len(axints)
            for j, axes_inter in enumerate(axints):
                chans[i]['volume']['axesInterpretation'][j] = {}
                for k, v in axes_inter.__dict__.items():
                    chans[i]['volume']['axesInterpretation'][j][str(k)] = str(v)
        out['channels'] = chans
        return out

    @classmethod
    def set_nested_dict_entry(cls, dict, keys, val):
        """Set a value (deep) in a nested dict.
        Args:
            dict : dict
                the nested dict
            keys : list
                the keys leading to the entry to set
            val : anything
                the value to set
        """
        currlvl = dict
        for i, key in enumerate(keys[:-1]):
            try:
                currlvl = currlvl[key]
            except KeyError:
                currlvl[key] = {}
                currlvl = currlvl[key]
        currlvl[keys[-1]] = val

    def __enter__(self):
        return self.nd2data

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):
        return self.get_frame(it)

    def __iter__(self):
        for i in range(self.sizes['T']):
            yield self[i]

    def __len__(self):
        return self.sizes['T']

    @property
    def shape(self):
        return self._shape

    def close(self):
        self.nd2data.close()

    def get_frame(self, index):
        """Load one frame of the movie
        Args:
            index : int
                the frame index to retrieve.
        Returns:
            frame : 2D array
                the image data of the frame
        """
        return self.nd2data[index]

    def tofile(self, file_handle, byte_order=None):
        raise NotImplementedError('Cannot write .nd2 file.')

    def camera_parameters(self, config):
        """get the camera specific parameters:
            * gain
            * quantum efficiency
            * wavelength
        These parameters depend on camera settings (as described in metadata)
        but the values themselves are given in the config.yaml file.
        Each filetype (nd2, ome-tiff, ..) has their own structure of metadata,
        which needs to be matched in the config.yaml description, as detailed
        in the specific child classes.

        The config file for the corresponding camera should look like this:
          Zyla 4.2:
            Pixelsize: 130
            Baseline: 100
            Quantum Efficiency:
              525: 0.7
              595: 0.72
              700: 0.64
            Sensitivity Categories:
              - PixelReadoutRate
              - ReadoutMode
            Sensitivity:
              540 MHz:
                Rolling Shutter at 16-bit: 7.18
              200 MHz:
                Rolling Shutter at 16-bit: 0.45
            Filter Wavelengths:
                1-R640: 700
                2-G561: 595
                3-B489: 525

        Args:
            config : dict
                description of camera parameters (for all possible settings)
        Returns:
            parameters : dict of lists of str
                keys: gain, qe, wavelength
        """
        parameters = {}
        info = self.meta

        try:
            assert "Cameras" in config.keys() and "Camera" in info.keys()
        except:
            # return {'gain': [1], 'qe': [1], 'wavelength': [0], 'cam_index': 0}
            raise KeyError("'camera' key not found in metadata or config.")

        cameras = config['Cameras']
        camera = info["Camera"]

        try:
            assert camera in cameras.keys()
        except:
            # return {'gain': [1], 'qe': [1], 'wavelength': [0], 'cam_index': 0}
            raise KeyError('camera from metadata not found in config.')

        index = sorted(list(cameras.keys())).index(camera)
        parameters['cam_index'] = index
        parameters['camera'] = camera

        try:
            assert "Picasso Metadata" in info
        except:
            return {'gain': [1], 'qe': [1], 'wavelength': [0], 'cam_index': 0}

        pm_info = info["Picasso Metadata"]
        cam_config = config["Cameras"][camera]
        if "Gain Property Name" in cam_config:
            raise NotImplementedError('extracting Gain from nd2 files is not implemented yet.')
            gain_property_name = cam_config["Gain Property Name"]
            gain = pm_info['gain']
            if "EM Switch Property" in cam_config:
                switch_property_name = cam_config[
                    "EM Switch Property"
                ]["Name"]
                switch_property_value = mm_info[
                    camera + "-" + switch_property_name
                ]
                if (
                    switch_property_value
                    == cam_config["EM Switch Property"][True]
                ):
                    parameters['gain'] = int(gain)
        if 'gain' not in parameters.keys():
            parameters['gain'] = [1]

        parameters['Sensitivity'] = {}
        if "Sensitivity Categories" in cam_config:
            categories = cam_config["Sensitivity Categories"]
            for i, category in enumerate(categories):
                parameters['Sensitivity'][category] = pm_info[category]
        if "Quantum Efficiency" in cam_config:
            if "Filter Wavelengths" in cam_config:
                channel = pm_info['Filter']
                channels = cam_config["Filter Wavelengths"]
                if channel in channels:
                    wavelength = channels[channel]
                    parameters['wavelength'] = str(wavelength)
                    parameters['qe'] = cam_config["Quantum Efficiency"][
                        wavelength]
        if 'qe' not in parameters.keys():
            parameters['qe'] = [1]
        if 'wavelength' not in parameters.keys():
            parameters['wavelength'] = [0]
        return parameters

    @property
    def dtype(self):
        return _np.dtype(self.meta['Data Type'])


class TiffMap:

    TIFF_TYPES = {1: "B", 2: "c", 3: "H", 4: "L", 5: "RATIONAL"}
    TYPE_SIZES = {
        "c": 1,
        "B": 1,
        "h": 2,
        "H": 2,
        "i": 4,
        "I": 4,
        "L": 4,
        "RATIONAL": 8,
    }

    def __init__(self, path, verbose=False):
        if verbose:
            print("Reading info from {}".format(path))
        self.path = _ospath.abspath(path)
        self.file = open(self.path, "rb")
        self._tif_byte_order = {b"II": "<", b"MM": ">"}[self.file.read(2)]
        self.file.seek(4)
        self.first_ifd_offset = self.read("L")

        # Read info from first IFD
        self.file.seek(self.first_ifd_offset)
        n_entries = self.read("H")
        for i in range(n_entries):
            self.file.seek(self.first_ifd_offset + 2 + i * 12)
            tag = self.read("H")
            type = self.TIFF_TYPES[self.read("H")]
            count = self.read("L")
            if tag == 256:
                self.width = self.read(type, count)
            elif tag == 257:
                self.height = self.read(type, count)
            elif tag == 258:
                bits_per_sample = self.read(type, count)
                dtype_str = "u" + str(int(bits_per_sample / 8))
                # Picasso uses internally exclusively little endian byte order
                self.dtype = _np.dtype(dtype_str)
                # the tif byte order might be different
                # so we also store the file dtype
                self._tif_dtype = _np.dtype(self._tif_byte_order + dtype_str)
        self.frame_shape = (self.height, self.width)
        self.frame_size = self.height * self.width

        # Collect image offsets
        self.image_offsets = []
        offset = self.first_ifd_offset
        while offset != 0:
            self.file.seek(offset)
            n_entries = self.read("H")
            if n_entries is None:
                # Some MM files have trailing nonsense bytes
                break
            for i in range(n_entries):
                self.file.seek(offset + 2 + i * 12)
                tag = self.read("H")
                if tag == 273:
                    type = self.TIFF_TYPES[self.read("H")]
                    count = self.read("L")
                    self.image_offsets.append(self.read(type, count))
                    break
            self.file.seek(offset + 2 + n_entries * 12)
            last_offset = offset + 2 + n_entries * 12
            offset = self.read("L")
        self.n_frames = len(self.image_offsets)
        self.last_ifd_offset = last_offset
        self.lock = _threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):

        with self.lock:  # for reading frames from multiple threads
            if isinstance(it, tuple):
                if isinstance(it, int) or _np.issubdtype(it[0], _np.integer):
                    return self[it[0]][it[1:]]
                elif isinstance(it[0], slice):
                    indices = range(*it[0].indices(self.n_frames))
                    stack = _np.array([self.get_frame(_) for _ in indices])
                    if len(indices) == 0:
                        return stack
                    else:
                        if len(it) == 2:
                            return stack[:, it[1]]
                        elif len(it) == 3:
                            return stack[:, it[1], it[2]]
                        else:
                            raise IndexError
                elif it[0] == Ellipsis:
                    stack = self[it[0]]
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            elif isinstance(it, slice):
                indices = range(*it.indices(self.n_frames))
                return _np.array([self.get_frame(_) for _ in indices])
            elif it == Ellipsis:
                return _np.array(
                    [self.get_frame(_) for _ in range(self.n_frames)]
                )
            elif isinstance(it, int) or _np.issubdtype(it, _np.integer):
                return self.get_frame(it)
            raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def info(self):
        info = {
            "Byte Order": self._tif_byte_order,
            "File": self.path,
            "Height": self.height,
            "Width": self.width,
            "Data Type": self.dtype.name,
            "Frames": self.n_frames,
        }
        # The following block is MM-specific
        self.file.seek(self.first_ifd_offset)
        n_entries = self.read("H")
        for i in range(n_entries):
            self.file.seek(self.first_ifd_offset + 2 + i * 12)
            tag = self.read("H")
            type = self.TIFF_TYPES[self.read("H")]
            count = self.read("L")
            if count * self.TYPE_SIZES[type] > 4:
                self.file.seek(self.read("L"))
            if tag == 51123:
                # This is the Micro-Manager tag
                # We generate an info dict that contains any info we need.
                readout = self.read(type, count).strip(
                    b"\0"
                )  # Strip null bytes which MM 1.4.22 adds
                mm_info_raw = _json.loads(readout.decode())
                # Convert to ensure compatbility with MM 2.0
                mm_info = {}
                for key in mm_info_raw.keys():
                    if key != "scopeDataKeys":
                        try:
                            mm_info[key] = mm_info_raw[key].get("PropVal")
                        except AttributeError:
                            mm_info[key] = mm_info_raw[key]

                info["Micro-Manager Metadata"] = mm_info
                if "Camera" in mm_info.keys():
                    info["Camera"] = mm_info["Camera"]
                else:
                    info["Camera"] = "None"
        # Acquistion comments
        self.file.seek(self.last_ifd_offset)
        comments = ""
        offset = 0
        while True:  # Fin the block with the summary
            line = self.file.readline()
            if "Summary" in str(line):
                break
            if not line:
                break
            offset += len(line)

        if line:
            for i in range(len(line)):
                self.file.seek(self.last_ifd_offset + offset + i)
                readout = self.read("L")
                if readout == 84720485:  # Acquisition comments
                    count = self.read("L")
                    readout = self.file.read(4 * count).strip(b"\0")
                    comments = _json.loads(readout.decode())["Summary"].split(
                        "\n"
                    )
                    break

        info["Micro-Manager Acquisiton Comments"] = comments

        return info

    def get_frame(self, index, array=None):
        self.file.seek(self.image_offsets[index])
        frame = _np.reshape(
            _np.fromfile(
                self.file, dtype=self._tif_dtype, count=self.frame_size
            ),
            self.frame_shape,
        )
        # We only want to deal with little endian byte order downstream:
        if self._tif_byte_order == ">":
            frame.byteswap(True)
            frame = frame.newbyteorder("<")
        return frame

    def read(self, type, count=1):
        if type == "c":
            return self.file.read(count)
        elif type == "RATIONAL":
            return self.read_numbers("L") / self.read_numbers("L")
        else:
            return self.read_numbers(type, count)

    def read_numbers(self, type, count=1):
        size = self.TYPE_SIZES[type]
        fmt = self._tif_byte_order + count * type
        try:
            return _struct.unpack(fmt, self.file.read(count * size))[0]
        except _struct.error:
            return None

    def close(self):
        self.file.close()

    def tofile(self, file_handle, byte_order=None):
        do_byteswap = byte_order != self.byte_order
        for image in self:
            if do_byteswap:
                image = image.byteswap()
            image.tofile(file_handle)


class TiffMultiMap(AbstractPicassoMovie):
    """Implments a subclass of AbstractPicassoMovie for reading
    ome tiff files created by MicroManager. Single files are
    maxed out at 4GB, so this class orchestrates reading from single files,
    accessed by TiffMap.
    """
    def __init__(self, path, memmap_frames=False, verbose=False):
        super().__init__()
        self.path = _ospath.abspath(path)
        self.dir = _ospath.dirname(self.path)
        base, ext = _ospath.splitext(
            _ospath.splitext(self.path)[0]
        )  # split two extensions as in .ome.tif
        base = _re.escape(base)
        pattern = _re.compile(
            base + r"_(\d*).ome.tif"
        )  # This matches the basename + an appendix of the file number
        entries = [_.path for _ in _os.scandir(self.dir) if _.is_file()]
        matches = [_re.match(pattern, _) for _ in entries]
        matches = [_ for _ in matches if _ is not None]
        paths_indices = [(int(_.group(1)), _.group(0)) for _ in matches]
        self.paths = [self.path] + [
            path for index, path in sorted(paths_indices)
        ]
        self.maps = [TiffMap(path, verbose=verbose) for path in self.paths]
        self.n_maps = len(self.maps)
        self.n_frames_per_map = [_.n_frames for _ in self.maps]
        self.n_frames = sum(self.n_frames_per_map)
        self.cum_n_frames = _np.insert(_np.cumsum(self.n_frames_per_map), 0, 0)
        self._dtype = self.maps[0].dtype
        self.height = self.maps[0].height
        self.width = self.maps[0].width
        self.shape = (self.n_frames, self.height, self.width)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):
        if isinstance(it, tuple):
            if it[0] == Ellipsis:
                stack = self[it[0]]
                if len(it) == 2:
                    return stack[:, it[1]]
                elif len(it) == 3:
                    return stack[:, it[1], it[2]]
                else:
                    raise IndexError
            elif isinstance(it[0], slice):
                indices = range(*it[0].indices(self.n_frames))
                stack = _np.array([self.get_frame(_) for _ in indices])
                if len(indices) == 0:
                    return stack
                else:
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            if isinstance(it[0], int) or _np.issubdtype(it[0], _np.integer):
                return self[it[0]][it[1:]]
        elif isinstance(it, slice):
            indices = range(*it.indices(self.n_frames))
            return _np.array([self.get_frame(_) for _ in indices])
        elif it == Ellipsis:
            return _np.array([self.get_frame(_) for _ in range(self.n_frames)])
        elif isinstance(it, int) or _np.issubdtype(it, _np.integer):
            return self.get_frame(it)
        raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def close(self):
        for map in self.maps:
            map.close()

    @property
    def dtype(self):
        return self._dtype

    def get_frame(self, index):
        # TODO deal with negative numbers
        for i in range(self.n_maps):
            if self.cum_n_frames[i] <= index < self.cum_n_frames[i + 1]:
                break
        else:
            raise IndexError
        return self.maps[i][index - self.cum_n_frames[i]]

    def info(self):
        info = self.maps[0].info()
        info["Frames"] = self.n_frames
        self.meta = info
        return info

    def camera_parameters(self, config):
        """Get the camera specific parameters:
            * gain
            * quantum efficiency
            * wavelength
        These parameters depend on camera settings (as described in metadata)
        but the values themselves are given in the config.yaml file.
        Each filetype (nd2, ome-tiff, ..) has their own structure of metadata,
        which needs to be matched in the config.yaml description, as detailed
        in the specific child classes.
        This code has been moved from localize to here, as it is file type
        specific (HG, April 2022).

        Args:
            config : dict
                description of camera parameters (for all possible settings)
        Returns:
            parameters : dict of lists of str
                keys: gain, qe, wavelength
        """
        # return {'gain': [1], 'qe': [1], 'wavelength': [0], 'cam_index': 0}
        parameters = {}
        info = self.meta

        try:
            assert "Cameras" in config and "Camera" in info
        except:
            return {'gain': [1], 'qe': [1], 'wavelength': [0], 'cam_index': 0}
            # raise KeyError("'camera' key not found in metadata or config.")

        cameras = config['Cameras']
        camera = info["Camera"]

        try:
            assert camera in list(cameras.keys())
        except:
            return {'gain': [1], 'qe': [1], 'wavelength': [0], 'cam_index': 0}
            # raise KeyError('camera from metadata not found in config.')

        index = sorted(list(cameras.keys())).index(camera)
        parameters['cam_index'] = index
        parameters['camera'] = camera

        try:
            assert "Micro-Manager Metadata" in info
        except:
            return {'gain': [1], 'qe': [1], 'wavelength': [0], 'cam_index': 0}

        mm_info = info["Micro-Manager Metadata"]
        cam_config = config["Cameras"][camera]
        if "Gain Property Name" in cam_config:
            gain_property_name = cam_config["Gain Property Name"]
            gain = mm_info[camera + "-" + gain_property_name]
            if "EM Switch Property" in cam_config:
                switch_property_name = cam_config[
                    "EM Switch Property"
                ]["Name"]
                switch_property_value = mm_info[
                    camera + "-" + switch_property_name
                ]
                if (
                    switch_property_value
                    == cam_config["EM Switch Property"][True]
                ):
                    parameters['gain'] = int(gain)
        if 'gain' not in parameters.keys():
            parameters['gain'] = [1]
        parameters['Sensitivity'] = {}
        if "Sensitivity Categories" in cam_config:
            categories = cam_config["Sensitivity Categories"]
            for i, category in enumerate(categories):
                property_name = camera + "-" + category
                if property_name in mm_info:
                    exp_setting = mm_info[camera + "-" + category]
                    parameters['Sensitivity'][category] = exp_setting
        if "Quantum Efficiency" in cam_config:
            if "Channel Device" in cam_config:
                channel_device_name = cam_config["Channel Device"][
                    "Name"
                ]
                channel = mm_info[channel_device_name]
                channels = cam_config["Channel Device"][
                    "Emission Wavelengths"
                ]
                if channel in channels:
                    wavelength = channels[channel]
                    parameters['wavelength'] = [str(wavelength)]
                    parameters['qe'] = [cam_config["Quantum Efficiency"][
                        wavelength]]
        if 'qe' not in parameters.keys():
            parameters['qe'] = [1]
        if 'wavelength' not in parameters.keys():
            parameters['wavelength'] = [0]
        return parameters

    def tofile(self, file_handle, byte_order=None):
        for map in self.maps:
            map.tofile(file_handle, byte_order)


def to_raw_combined(basename, paths):
    raw_file_name = basename + ".ome.raw"
    with open(raw_file_name, "wb") as file_handle:
        with TiffMap(paths[0]) as tif:
            tif.tofile(file_handle, "<")
            info = tif.info()
        for path in paths[1:]:
            with TiffMap(path) as tif:
                info_ = tif.info()
                info["Frames"] += info_["Frames"]
                if "Comments" in info_:
                    info["Comments"] = info_["Comments"]
                tif.tofile(file_handle, "<")
        info["Generated by"] = "Picasso ToRaw"
        info["Byte Order"] = "<"
        info["Original File"] = _ospath.basename(info.pop("File"))
        info["Raw File"] = _ospath.basename(raw_file_name)
        save_info(basename + ".ome.yaml", [info])


def get_movie_groups(paths):
    groups = {}
    if len(paths) > 0:
        pattern = _re.compile(
            r"(.*?)(_(\d*))?.ome.tif"
        )  # This matches the basename + an opt appendix of the file number
        matches = [_re.match(pattern, path) for path in paths]
        match_infos = [
            {"path": _.group(), "base": _.group(1), "index": _.group(3)}
            for _ in matches
        ]
        for match_info in match_infos:
            if match_info["index"] is None:
                match_info["index"] = 0
            else:
                match_info["index"] = int(match_info["index"])
        basenames = set([_["base"] for _ in match_infos])
        for basename in basenames:
            match_infos_group = [
                _ for _ in match_infos if _["base"] == basename
            ]
            group = [_["path"] for _ in match_infos_group]
            indices = [_["index"] for _ in match_infos_group]
            group = [path for (index, path) in sorted(zip(indices, group))]
            groups[basename] = group
    return groups


def to_raw(path, verbose=True):
    paths = _glob.glob(path)
    groups = get_movie_groups(paths)
    n_groups = len(groups)
    if n_groups:
        for i, (basename, group) in enumerate(groups.items()):
            if verbose:
                print(
                    "Converting movie {}/{}...".format(i + 1, n_groups),
                    end="\r",
                )
            to_raw_combined(basename, group)
        if verbose:
            print()
    else:
        if verbose:
            print("No files matching {}".format(path))


def save_datasets(path, info, **kwargs):
    with _h5py.File(path, "w") as hdf:
        for key, val in kwargs.items():
            hdf.create_dataset(key, data=val)
    base, ext = _ospath.splitext(path)
    info_path = base + ".yaml"
    save_info(info_path, info)


def save_locs(path, locs, info):
    locs = _lib.ensure_sanity(locs, info)
    with _h5py.File(path, "w") as locs_file:
        locs_file.create_dataset("locs", data=locs)
    base, ext = _ospath.splitext(path)
    info_path = base + ".yaml"
    save_info(info_path, info)


def load_locs(path, qt_parent=None):
    with _h5py.File(path, "r") as locs_file:
        locs = locs_file["locs"][...]
    locs = _np.rec.array(
        locs, dtype=locs.dtype
    )  # Convert to rec array with fields as attributes
    info = load_info(path, qt_parent=qt_parent)
    return locs, info


def load_clusters(path, qt_parent=None):
    with _h5py.File(path, "r") as cluster_file:
        clusters = cluster_file["clusters"][...]
    clusters = _np.rec.array(
        clusters, dtype=clusters.dtype
    )  # Convert to rec array with fields as attributes
    return clusters


def load_filter(path, qt_parent=None):
    with _h5py.File(path, "r") as locs_file:
        try:
            locs = locs_file["locs"][...]
            info = load_info(path, qt_parent=qt_parent)
        except KeyError:
            try:
                locs = locs_file["groups"][...]
                info = load_info(path, qt_parent=qt_parent)
            except KeyError:
                locs = locs_file["clusters"][...]
                info = []

    locs = _np.rec.array(
        locs, dtype=locs.dtype
    )  # Convert to rec array with fields as attributes
    return locs, info
