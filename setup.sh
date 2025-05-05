conda create -n skylark python=3.11
conda activate skylark
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt

pushd model\segment_anything_2
python setup.py build_ext --inplace
popd

# python inference.py --version YxZhang/evf-sam2 --precision='fp16' --vis_save_path "vis" --model_type sam2 --image_path "assets/zebra.jpg" --prompt "zebra top left"