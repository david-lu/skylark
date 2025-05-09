pyenv global 3.11
python -m venv .venv
source .venv/bin/activate

sudo apt install -y libffi-dev build-essential zlib1g-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libncurses-dev libncursesw5-dev xz-utils tk-dev
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

pushd model/segment_anything_2
python setup.py build_ext --inplace
popd

# python inference.py --version YxZhang/evf-sam2 --precision='fp16' --vis_save_path "vis" --model_type sam2 --image_path "assets/zebra.jpg" --prompt "zebra top left"   