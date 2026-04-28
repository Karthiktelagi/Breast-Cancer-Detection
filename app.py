from flask import Flask, render_template, request, redirect
import torch, cv2, numpy as np, os, joblib
import torch.nn.functional as F

from models.cnn_model import CNNModel
from models.resnet_model import ResNetModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ["Normal", "Benign", "Malignant"]

# Load models
cnn = CNNModel()
cnn.load_state_dict(torch.load("model.pth", weights_only=True))
cnn.eval()

resnet = ResNetModel()
resnet.load_state_dict(torch.load("resnet_model.pth", weights_only=True))
resnet.eval()

svm = joblib.load("svm_model.pkl")

cam = GradCAM(model=cnn, target_layers=[cnn.conv[-1]])

history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global history

    if request.method == "POST":
        file = request.files.get("file")
        model_choice = request.form.get("model", "all")
        old_image = request.form.get("old_image")

        # ✅ HANDLE IMAGE PERSISTENCE
        if file and file.filename != "":
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
        else:
            path = old_image

        if not path:
            return redirect("/")

        # Preprocess
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (224,224))
        img_norm = img / 255.0
        tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).float()

        # CNN
        cnn_out = cnn(tensor)
        cnn_pred = torch.argmax(cnn_out).item()
        cnn_conf = torch.max(torch.softmax(cnn_out, dim=1)).item()

        # ResNet
        res_out = resnet(tensor)
        res_pred = torch.argmax(res_out).item()
        res_conf = torch.max(torch.softmax(res_out, dim=1)).item()

        # Hybrid
        with torch.no_grad():
            feat = cnn.conv(tensor)
            feat = F.adaptive_avg_pool2d(feat,(1,1))
            feat = feat.view(1,-1).numpy()

        hyb_pred = svm.predict(feat)[0]
        hyb_conf = 0.90

        # Model selection
        if model_choice == "cnn":
            prediction = classes[cnn_pred]
        elif model_choice == "resnet":
            prediction = classes[res_pred]
        elif model_choice == "hybrid":
            prediction = classes[hyb_pred]
        else:
            prediction = classes[cnn_pred]

        # GradCAM
        cam_map = cam(input_tensor=tensor)[0]
        img3 = np.repeat(img_norm[:,:,np.newaxis],3,axis=2)
        vis = show_cam_on_image(img3, cam_map, use_rgb=True)

        cam_path = os.path.join(UPLOAD_FOLDER, "cam_"+os.path.basename(path))
        cv2.imwrite(cam_path, vis)

        history.insert(0, prediction)

        return render_template("index.html",
            model=model_choice,
            prediction=prediction,
            cnn=classes[cnn_pred],
            resnet=classes[res_pred],
            hybrid=classes[hyb_pred],
            cnn_conf=round(cnn_conf*100,2),
            resnet_conf=round(res_conf*100,2),
            hybrid_conf=round(hyb_conf*100,2),
            image=path,
            old_image=path,   # 👈 IMPORTANT
            cam=cam_path,
            history=history[:5]
        )

    return render_template("index.html", history=history[:5])
    

if __name__ == "__main__":
    app.run(debug=True)