from PIL import Image
from transformers import pipeline


class EmotionRecognition:
    def __init__(self):
        self.model = pipeline("image-classification",
                              model="kdhht2334/autotrain-diffusion-emotion-facial-expression-recognition-40429105176", device=0, framework="pt")

    def run(self, image, is_dir=False):
        if is_dir:
            image = Image.open(image)
        res = self.model(image)
        emotions = self._parse_emotions(res)
        return self._determine_dominant_emotion(emotions)

    @staticmethod
    def _parse_emotions(res):
        emotions = {}
        if res:
            for e in res:
                emotions[e["label"]] = e["score"]
        return emotions

    @staticmethod
    def _determine_dominant_emotion(emotions):
        if not emotions:
            return None

        if "neutral" in emotions:
            if emotions["neutral"] == max(emotions.values()):
                return "neutral"

        return max(emotions, key=emotions.get)