"""Technical skill vocabulary and alias-aware extraction."""

import re


TECH_SKILLS = [
    "python", "java", "javascript", "typescript", "sql", "r", "c++", "c#", "c", "go", "rust", "scala",
    "tensorflow", "keras", "pytorch", "scikit-learn", "sklearn", "xgboost", "lightgbm",
    "nlp", "bert", "gpt", "t5", "transformers", "spacy", "nltk", "word2vec", "glove",
    "fasttext", "tfidf", "tf-idf", "word embeddings", "sentence transformers", "langchain",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau", "power bi",
    "machine learning", "deep learning", "neural networks", "computer vision",
    "random forest", "logistic regression", "svm", "naive bayes", "gradient boosting",
    "classification", "regression", "clustering", "recommendation", "reinforcement learning",
    "git", "github", "docker", "kubernetes", "aws", "gcp", "azure", "mlflow", "airflow",
    "flask", "fastapi", "streamlit", "jupyter", "linux", "node.js", ".net",
    "mysql", "postgresql", "mongodb", "redis", "sqlite",
    "speech recognition", "text to speech", "tts", "stt", "gtts", "pyttsx3",
    "api", "rest api", "opencv", "hadoop", "spark", "data visualization",
    "prompt engineering", "openai", "hugging face", "data mining", "statistics",
    "data analysis", "feature engineering", "model deployment",
]

SKILL_ALIASES = {
    "js": "javascript",
    "ecmascript": "javascript",
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "nodejs": "node.js",
    "node js": "node.js",
    "dotnet": ".net",
    "c sharp": "c#",
    "c-sharp": "c#",
    "cpp": "c++",
    "c plus plus": "c++",
}


def _contains_skill(text: str, skill: str) -> bool:
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(skill) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_skills(text: str) -> list[str]:
    normalized = text.lower()
    found = {skill for skill in TECH_SKILLS if _contains_skill(normalized, skill)}
    for alias, canonical in SKILL_ALIASES.items():
        if _contains_skill(normalized, alias):
            found.add(canonical)
    return [skill for skill in TECH_SKILLS if skill in found]