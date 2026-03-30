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
        file = request.files["file"]

        if file.filename == "":
            return redirect("/")

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

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
        hyb_conf = 0.90  # approx

        # Final decision (CNN)
        final = classes[cnn_pred]

        # GradCAM
        cam_map = cam(input_tensor=tensor)[0]
        img3 = np.repeat(img_norm[:,:,np.newaxis],3,axis=2)
        vis = show_cam_on_image(img3, cam_map, use_rgb=True)

        cam_path = os.path.join(UPLOAD_FOLDER, "cam_"+file.filename)
        cv2.imwrite(cam_path, vis)

        # Save history
        history.insert(0, final)

        return render_template("index.html",
            cnn=classes[cnn_pred],
            resnet=classes[res_pred],
            hybrid=classes[hyb_pred],
            cnn_conf=round(cnn_conf*100,2),
            res_conf=round(res_conf*100,2),
            hyb_conf=round(hyb_conf*100,2),
            final=final,
            image=path,
            cam=cam_path,
            history=history[:5]
        )

    return render_template("index.html", history=history[:5])

if __name__ == "__main__":
    app.run(debug=True)