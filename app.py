import gradio as gr
from fastai.vision.all import *

# Load the pre-trained model
learn = load_learner('export.pkl')

def predict_image(img):
    """
    Predicts the label of an input image using the loaded fastai learner.
    """
    # Gradio passes a numpy array, fastai expects a PILImage or similar
    # We can convert it to a PILImage or let fastai handle the conversion if it's compatible
    # For simplicity, let's assume the learner can handle the input directly or convert it.
    # If not, you might need to convert img to PILImage: Image.fromarray(img)
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=3),
    title="Image Classifier",
    description="Upload an image to get its predicted label."
)

demo.launch()
