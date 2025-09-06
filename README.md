# phonikud-byt5

Hebrew G2P with byt5 model based on [Phonikud](https://phonikud.github.io)

## Prepare

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
sudo apt install -y p7zip-full wget
wget https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z
7z x knesset_phonemes_v1.txt.7z
mkdir data
head -n 2000000 knesset_phonemes_v1.txt > ./data/data.txt
```

Unvocalized data.txt

```console
uv run python -c "import re, pathlib; p=pathlib.Path('data/data.txt'); txt=p.read_text(encoding='utf-8'); txt=re.sub(r'[|\u0590-\u05cf]', '', txt); pathlib.Path('data/data_clean.txt').write_text(txt, encoding='utf-8')"
rm -rf ./data/data.txt
head ./data/data_clean.txt
```

```console
uv run src/phonikud_byt5/run_train.py
```

## Logs

```console
uv run wandb login
```

## Onnx

See [byt5-onnx](./byt5-onnx/)