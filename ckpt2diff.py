import importlib.util
import io
import os
import traceback
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import IO, Callable

import click
import patch_ng as patch
import requests

MODELS_DIR: str = "models/"
CKPT_FILE: str = MODELS_DIR + "model.ckpt"
CONFIG_FILE: str = MODELS_DIR + "config.yaml"
HF_MODEL_DIR: str = MODELS_DIR + "diffusers_model"

DEFAULT_CONFIG_URL: str = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-inference.yaml"
CONVERTER_SCRIPT_URL: str = "https://huggingface.co/spaces/diffusers/convert-sd-ckpt/resolve/main/convert.py"
CONVERTER_PATCH_DIFF: bytes = b"""
--- convert.py
+++ convert.py
@@ -785,7 +785,11 @@
 
     for key in keys:
         if key.startswith("cond_stage_model.transformer"):
-            text_model_dict[key[len("cond_stage_model.transformer.") :]] = checkpoint[
+            # patched here to support non-standard models
+            mapped_key = key[len("cond_stage_model.transformer.") :]
+            if not mapped_key.startswith("text_model"):
+                mapped_key = "text_model." + mapped_key
+            text_model_dict[mapped_key] = checkpoint[
                 key
             ]
 
"""

"""
This user-friendly wizard is used to convert a Stable Diffusion Model from CKPT format to Diffusers format.
"""


def load_converter(script_url: str, patch_diff: bytes) -> ModuleType:
    response = requests.get(script_url)
    response.raise_for_status()
    source: str = response.text

    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "convert.py"), "w") as fp:
            fp.write(source)
        pset = patch.fromstring(patch_diff)

        # rewrite chdir because patch-ng code is unsafe
        cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            if not pset.apply():
                raise RuntimeError("patching failed")
        finally:
            os.chdir(cwd)
        del cwd

        with open(os.path.join(tmpdir, "convert.py"), "r") as fp:
            source = fp.read()

    spec = importlib.util.spec_from_loader("converter", None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    exec(source, module.__dict__)

    return module


def load_config(config_path: str, default_config_url: str) -> IO:
    config_path = config_path.strip()

    if os.path.exists(config_path):
        return open(config_path, "r")
    else:
        response = requests.get(default_config_url)
        response.raise_for_status()
        return io.BytesIO(response.content)


def convert(converter: ModuleType, ckpt_path: str, config_fp: IO, scheduler_type: str, extract_ema: bool,
            hf_model_path: str) -> None:
    convert_full_checkpoint = getattr(converter, "convert_full_checkpoint", None)
    if callable(convert_full_checkpoint):
        convert_full_checkpoint(
            ckpt_path,
            config_fp,
            scheduler_type=scheduler_type,
            extract_ema=extract_ema,
            output_path=hf_model_path,
        )
    else:
        raise AttributeError("missing required entry function")


@click.command()
def wizard():
    click.echo(click.style("", fg="bright_green"), nl=False)

    click.echo("Welcome to Ckpt2Diff, modified from HuggingFace App diffusers/convert-sd-ckpt")
    click.echo("This wizard will help you to convert a Stable Diffusion Model "
               "from CKPT format to Diffusers format.")
    click.echo("You can find the latest source code at https://github.com/Sunbread/ckpt2diff")
    click.echo()
    click.echo("Make sure to activate virtualenv and install all dependencies.")
    click.echo("Run \"pip install -r requirements.txt\" to install.")
    click.echo()
    click.echo(f"Please put your CKPT file into {CKPT_FILE}")
    click.echo(f"and your config file into {CONFIG_FILE} if you have,")
    click.echo(f"and the result will be in {HF_MODEL_DIR}.")
    click.echo()
    click.confirm("Ready to begin?", abort=True)

    click.echo("Downloading and patching the latest converter from HuggingFace...")
    try:
        converter: ModuleType = load_converter(CONVERTER_SCRIPT_URL, CONVERTER_PATCH_DIFF)
    except Exception:
        click.echo("Oops, something went wrong while loading the converter.")
        traceback.print_exc()
        raise click.Abort
    click.echo("Done.")

    scheduler: str = click.prompt(
        "Choose Scheduler Type",
        type=click.Choice(["PNDM", "K-LMS", "Euler", "EulerAncestral", "DDIM"], case_sensitive=False),
        default="PNDM",
    )
    ema: bool = click.confirm(
        "Extract EMA",
        default=True,
    )
    if not os.path.exists(CKPT_FILE):
        click.echo(f"Checkpoint file {CKPT_FILE} does not exist. I told you DO NOT move the CKPT file!")
        raise click.Abort
    if not os.path.exists(CONFIG_FILE):
        click.confirm(
            f"Config file {CONFIG_FILE} does not exist and default config on url {DEFAULT_CONFIG_URL} will be used. \n"
            "Do you want to continue?",
            abort=True,
        )

    click.echo("Invoke converter to perform conversion.")
    click.echo()
    convert(converter, CKPT_FILE, load_config(CONFIG_FILE, DEFAULT_CONFIG_URL), scheduler, ema, HF_MODEL_DIR)
    click.echo()
    click.echo("Done!")


if __name__ == "__main__":
    wizard()
