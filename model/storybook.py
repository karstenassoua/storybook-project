# storybook_test.py
import os
from pathlib import Path
from openai import OpenAI


def _load_openai_key_from_dotenv() -> None:
    """Load OPENAI_API_KEY from a local .env file if not already set."""
    if os.getenv("OPENAI_API_KEY"):
        return

    # Look for .env in project root and current working directory.
    candidate_paths = [
        Path(__file__).resolve().parents[1] / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in candidate_paths:
        if not env_path.exists():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
                return


_load_openai_key_from_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set it in your shell or create a .env file at project root "
        "(/Users/kass/Desktop/MML/Storybook Project/.env) with: OPENAI_API_KEY=your-key"
    )

# Uses OPENAI_API_KEY from environment or local .env
client = OpenAI()

model_id = "ft:gpt-4o-mini-2024-07-18:personal:storybook1:D3j5j130"

resp = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "user", "content": """You are a children's book content analyzer. Your task is to read the book text and assign the output one of nine lesson codes based on it's primary moral, educational, or thematic takeaway. 
Read the entire Book Text carefully and determine which single lesson code best represents the main message or value that a child reader would take away from the story. Even if multiple themes are present, choose the one that is most central to the narrative.
THE NINE LESSON CODES:
FAM - Love of Family: The Book Text centers relationships between nuclear family members, depicting this love as special and unconditional. The reader learns to value their parents or other family members. Example: A story where parents take care of their child.
FRI - Friendship The Book Text features strong friendships between characters. Friendships may be tested but friends emerge stronger. The focus is on loyalty, reconciliation, or mutual care between friends. The reader learns the importance of having and maintaining friendships with care and reciprocity. Example: Two friends argue but work together to reconcile. Note: If the book primarily emphasizes loyalty, reconciliation, or mutual care between friends, especially if the moral centers on maintaining or valuing friendship, use FRI not JUS.
REL - Religion: The Book Text features overtly religious sentiments, actions, or quotations, often linking religious belief with prosociality. The reader learns the value of religious faith. Example: A protagonist prays for their friends and family.
EMO - Emotions: The Book Text guides readers toward recognizing, expressing, and managing emotions and their effects. Characters experience emotional changes throughout the story. May include characters coping with self-doubt who come to accept themselves. The reader learns that emotions are temporary and manageable. Example: A protagonist is angry about not getting their way but later accepts it and becomes content with what they have.
ADV - Adventure/Trying New Things: The Book Text features characters moving through different locations or settings as part of their journey. The narrative highlights exploration, venturing outside one's comfort zone, and embracing new experiences. Characters are often rewarded for bravery. The reader learns that exploring new things can lead to positive experiences. Example: A protagonist moves to a new school and learns to embrace the environment change. Note: The presence of an unfamiliar situation is necessary. A significant challenge is not required (that would be PER).
PER - Perseverance: The Book Text  features characters facing situational challenges such as obstacles, setbacks, or difficult conditions. The reader learns the value of effort and hard work, especially through difficulty. Example: A protagonist improves their grades in school through hard work and studying. Note: The character must apply effort or face difficulty. An unfamiliar situation is not required (that would be ADV).
JUS - Justice and Fairness: The Book Text emphasizes fairness or moral consequences, whether grand scale (heroes vs. villains, good triumphing over evil) or everyday contexts (sharing, kindness, fair treatment). Right actions are rewarded and wrong actions bring penalties. Example: A protagonist shares with a stranger who later is able to give them something in return.
NAT - The Natural World: The Book Text familiarizes readers with and conveys value in animal or plant life. The reader learns that non-human animals and plant life are bearers of value. Example: A protagonist takes great care to preserve a garden. Note: Many books feature anthropomorphized animal protagonists. The mere presence of animal characters is not sufficient. The book must convey value in animals or plants beyond their exhibition of human traits.
NUL - Books Without a Lesson: The Book Text is designed primarily for fun, entertainment, or comfort, such as silly stories or bedtime routines, without an explicit moral, educational takeaway, or skill-building element. Readers read these books for reading's sake. Example: A protagonist plays hide and seek with the reader.
INSTRUCTIONS:
Read the complete book text provided
Identify the primary lesson or takeaway a child reader would receive
Select the single code that best matches this primary lesson
Output only the three-letter code
OUTPUT FORMAT: Return the three-letter code then a 1 sentence rationale as to why your prediction is the primary theme for that book. Example: "NUL - There is no primary lesson for this book that aligns with the theme categories."""},
        {"role": "user", "content": """A Lion lay asleep in the forest, his great head resting on his paws. A timid little Mouse came upon him unexpectedly, and in her fright and haste to get away, ran across the Lion's nose. Roused from his nap, the Lion laid his huge paw angrily on the tiny creature to kill her.

"Spare me!" begged the poor Mouse. "Please let me go and some day I will surely repay you."

The Lion was much amused to think that a Mouse could ever help him. But he was generous and finally let the Mouse go.

Some days later, while stalking his prey in the forest, the Lion was caught in the toils of a hunter's net. Unable to free himself, he filled the forest with his angry roaring. The Mouse knew the voice and quickly found the Lion struggling in the net. Running to one of the great ropes that bound him, she gnawed it until it parted, and soon the Lion was free.

"You laughed when I said I would repay you," said the Mouse. "Now you see that even a Mouse can help a Lion."""}
    ]
)

print(resp.choices[0].message.content)
