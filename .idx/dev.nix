# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-23.11"; # or "unstable"
  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
  ];
  # Sets environment variables in the workspace
  env = {};
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "ms-toolsai.jupyter"
      "ms-python.python"
    ];
    workspace = {
      onCreate = {
        create-venv = ''
          python -m venv .venv --without-pip
          .venv/bin/python -m ensurepip
          .venv/bin/python -m pip install --upgrade pip setuptools wheel
          if [ -f requirements.txt ]; then
            .venv/bin/pip install -r requirements.txt
          fi
        '';
      };
      onStart = {
        activate-venv = "source .venv/bin/activate";
      };
    };

    # Enable previews and customize configuration
    previews = {};
  };
}
