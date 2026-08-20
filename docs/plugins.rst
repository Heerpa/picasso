=======
Plugins
=======

Usage
-----
A plugin is a single Python (``.py``) file that adds new actions to a Picasso GUI.

**NOTE**: A plugin is ordinary Python code running inside Picasso, so an enabled plugin can do anything Picasso itself can do: read and write your files, use the network, start other programs. Picasso cannot sandbox plugins.

- Plugins are **not loaded until you enable them**, so copying a file into the plugins folder is never enough to make it run;
- Downloads from the online registry are **checked against the SHA-256 hash published in the registry**, so the code behind a plugin cannot change without a visible, reviewed change to the registry;
- The plugin browser can **show you a plugin's source** before you install it, and a **diff of what changed** before you update it.

Installing plugins from the online registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The easiest way to get plugins is the built-in store: open **Plugins → Browse online plugins…** in any Picasso app. This lists the plugins published in the `picasso_plugins registry <https://github.com/jungmannlab/picasso_plugins>`_, filtered to the app you are using (tick *Show plugins for all Picasso apps* to see the rest), and lets you **Install**, **Update** or **Uninstall** each one with a single click. Installed plugins take effect immediately — no restart needed.

Each row also offers **View source** (**View changes** for an update), which downloads the file, verifies it against the registry hash and shows it read-only. Most plugins are short; reading one before enabling it is realistic and is the only check that actually tells you what a plugin does.

A plugin you install this way is enabled automatically — clicking *Install* and confirming the warning *is* the consent step. Picasso shows that warning once, before your first install.

If a plugin is listed as **No integrity hash**, the registry does not publish a SHA-256 for it and Picasso will refuse to install it. That is a problem with the registry entry, not with your setup; please report it in the `picasso_plugins repository <https://github.com/jungmannlab/picasso_plugins>`_.

Installing plugins manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Place your plugin ``.py`` file(s) in the user plugins folder:

- ``~/.picasso/plugins`` (on Windows this is ``C:\Users\<your user name>\.picasso\plugins``).

This folder is created automatically the first time you run any Picasso GUI, and the **same folder is used for every installation type** (one-click installer, PyPI, conda or GitHub). The easiest way to open it is the **Plugins → Open plugins folder…** menu entry available inside any Picasso app.

A file copied in by hand is **found but not enabled**. It shows up in **Plugins → Browse online plugins…** as a *Local file*, and the Plugins menu tells you how many such files are waiting. To start using it, review it with **View source**, then tick its **Enabled** checkbox and confirm the warning. Untick the checkbox at any time to stop loading it without deleting the file.

Because Picasso cannot verify a file that did not come from the registry, this is the least safe way to share a plugin. If you are sharing your own plugin with others, publishing it in the `registry <https://github.com/jungmannlab/picasso_plugins>`_ is better for everyone: recipients get a hash-verified download, a version number and one-click updates, and the code is reviewed in the open before it reaches anyone.

Because the plugins folder lives in your home directory and not inside the Picasso installation, plugins are kept when you update Picasso and are not left behind when you uninstall it.

**NOTE**: With the one-click installer, plugins can only use packages that are installed with Picasso (the dependencies listed in ``pyproject.toml``).

For developers
--------------
To create a plugin, you can use the template provided in `picasso/plugin_template.py <https://github.com/jungmannlab/picasso/blob/master/plugin_template.py>`_. For more examples of plugins, please see the `GitHub repo <https://github.com/jungmannlab/picasso_plugins>`_.

Registry entries
~~~~~~~~~~~~~~~~
Every entry in the registry's ``index.json`` must carry the SHA-256 of the file it points to, or Picasso will not install it::

    {
      "id": "render_extras",
      "display_name": "Render extras",
      "app": "render",
      "description": "Extra export actions for Render.",
      "author": "Jungmann Lab",
      "version": "1.0.0",
      "file": "plugins/render_extras.py",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "min_picasso_version": "0.11.0"
    }

``id`` must consist of letters, digits, ``_`` and ``-`` only (it becomes the local file name), and ``file`` must be a relative path to a ``.py`` file inside the registry repository. Entries that violate either rule are ignored. Whenever you change a plugin file, update both ``version`` and ``sha256`` in the same commit — the manifest is what pins the code, so the hash is what makes the change reviewable.

Note that plugin state (which files are installed and which are enabled) lives in a hidden ``.installed.json`` sidecar in the plugins folder.
