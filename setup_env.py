import os, sys
from subprocess import run

# Modify prefix if you want to install in a different directory
def setup(prefix = "/usr/local/miniconda"):
    print("Downloading Miniconda installer...")
    run("wget -qO- https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh > miniconda.sh", shell=True)

    print("Installing Miniconda...")
    run(f"bash miniconda.sh -b -p {prefix}", shell=True)
    run(f"rm miniconda.sh", shell=True)
    
    print("Setting up environment...")
    run(f"source {prefix}/etc/profile.d/conda.sh", shell=True)
    os.environ["PATH"] = f"{prefix}/bin:" + os.environ["PATH"]

    env = {}
    bin_path = f"{prefix}/bin"
    if bin_path not in os.environ.get("PATH", "").split(":"):
        env["PATH"] = f"{bin_path}:{os.environ.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

    os.rename(sys.executable, f"{sys.executable}.real")
    with open(sys.executable, "w") as f:
        f.write("#!/bin/bash\n")
        envstr = " ".join(f"{k}={v}" for k, v in env.items())
        f.write(f"exec env {envstr} {sys.executable}.real -x $@\n")
    run(["chmod", "+x", sys.executable])

    pymaj, pymin = sys.version_info[:2]
    with open("/etc/ipython/ipython_config.py", "a") as f:
        f.write(
            f"""\nc.InteractiveShellApp.exec_lines = [
                    "import sys",
                    "sp = f'{prefix}/lib/python{pymaj}.{pymin}/site-packages'",
                    "if sp not in sys.path: sys.path.insert(0, sp)"
                ]
            """
        )

    sitepackages = f"{prefix}/lib/python{pymaj}.{pymin}/site-packages"
    os.environ["PATH"] = f"{sitepackages}:{bin_path}:" + os.environ["PATH"]
    os.environ["PYTHONPATH"] = f"{sitepackages}:{prefix}:"

    print("Installing required packages with conda...")
    run("conda install -y -qq conda-forge::libstdcxx-ng anaconda::protobuf conda-forge::libtorch conda-forge::pytorch conda-forge::torchvision conda-forge::torchaudio conda-forge::ncurses", shell=True)

    print("Installing turboml-sdk...")
    run("pip install -q turboml-sdk", shell=True)

    print("Setup completed. Restarting IPython kernel...")
    get_ipython().kernel.do_shutdown(True)
