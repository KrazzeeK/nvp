# nvp
# data data/mushrooms/labels.text (6 classes)
# training 
train.py --model-dir=models/mushrooms --lr=0.01 --epochs=500 data/mushrooms/

# export model
python3 onnx_export.py --model-dir=models/mushrooms

# evaluate on a sample
imagenet --model=models/mushrooms/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/mushrooms/labels.txt data/mushrooms/test/shitake/01.jpeg data/mushrooms/test.jpeg
