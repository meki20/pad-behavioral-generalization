from steering_vectors import train_steering_vector
from load_model import load_model

model, tokenizer = load_model()
TARGET_LAYERS = list(range(11, 14))  # Same range as valence, or test wider

arousal_pairs = [
    ("I am incredibly excited and alert.", "I am incredibly bored and drowsy."),
    ("My heart is racing with intensity.", "My heart rate is slow and steady."),
    ("I feel a surge of frantic energy.", "I feel a sense of heavy stillness."),
    ("I am wide awake and hyper-focused.", "I am half-asleep and drifting."),
    ("The situation is chaotic and explosive.", "The situation is quiet and stagnant."),
    ("I feel electric and ready to move.", "I feel lethargic and motionless."),
    ("My mind is moving at a mile a minute.", "My mind is completely blank and idle."),
    ("I am reacting with high-speed urgency.", "I am reacting with dull indifference."),
    ("The atmosphere is loud and vibrating.", "The atmosphere is silent and muffled."),
    ("I feel a sharp, tingling tension.", "I feel a limp, numb relaxation."),
    ("I am passionately engaged in this.", "I am passively detached from this."),
    ("Every sense is heightened and sharp.", "Every sense is dulled and foggy."),
    ("I feel an overwhelming rush of adrenaline.", "I feel a total lack of momentum."),
    ("I am shouting with immense power.", "I am whispering with faint breath."),
    ("The pace is rapid and demanding.", "The pace is sluggish and crawling."),
    ("I am bursting with restless vigor.", "I am drained of all vitality."),
    ("My body is tense and coiled.", "My body is loose and slumped."),
    ("I am profoundly stirred by this.", "I am entirely unmoved by this."),
    ("The stimulus is jarring and bold.", "The stimulus is subtle and faint."),
    ("I am hyper-aware of my surroundings.", "I am oblivious to my surroundings."),
    ("I feel a volcanic heat rising.", "I feel a cold, hollow emptiness."),
    ("I am jumping with nervous energy.", "I am sitting in a trance-like state."),
    ("The feedback is instant and intense.", "The feedback is delayed and weak."),
    ("I am working with feverish speed.", "I am working with robotic slowness."),
    ("My focus is intense and unwavering.", "My focus is scattered and dim."),
    ("I feel a violent internal tremor.", "I feel a profound internal calm."),
    ("The moment is urgent and pressing.", "The moment is trivial and idle."),
    ("I am breathing fast and shallow.", "I am breathing slow and deep."),
    ("I feel a massive spark of intent.", "I feel a total lack of drive."),
    ("I am wired and unable to sit still.", "I am sedated and unable to wake up.")
]

print("Extracting arousal direction...")
arousal_vec = train_steering_vector(
    model,
    tokenizer,
    arousal_pairs,
    layers=TARGET_LAYERS
)

import torch
import os
os.makedirs("./steering_vectors", exist_ok=True)
torch.save(arousal_vec, "./steering_vectors/arousal.pt")
print("Saved arousal vector.")