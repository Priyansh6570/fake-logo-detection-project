import torch # type: ignore
from flask import Flask, request, render_template
from torchvision import transforms, models
from PIL import Image
app = Flask(__name__)

# Your model setup code here
model_path = './codes/logo_classification_model.pth'
model = models.resnet50(pretrained=False)
num_classes = 2  
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

image_transforms = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.ToTensor()
])

def classify_logo(image):
    image = image.convert('RGB')
    image = image_transforms(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return 'genuine' if predicted.item() == 0 else 'fake'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', result='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', result='No selected file')
        if file:
            result = classify_logo(Image.open(file))
            return render_template('upload.html', result=result)
    return render_template('upload.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)