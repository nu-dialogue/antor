import setuptools

requirements = [
    "scikit-learn>=0.22.0",
    "torch==1.7.1",
    "transformers>=4.17.0",
    "jupyter",
    "wandb",
    "pandas",
    "seaborn",
    "google-cloud-speech",
    "google-cloud-texttospeech",
    "audiomentations[extras]",
    "edit-distance",
    "jiwer"
]

setuptools.setup(
    name="antor",
    version="0.0.1",
    packages=setuptools.find_packages(),
    author="ohashi56225",
    author_email="ohashi.atsumoto@g.mbox.nagoya-u.ac.jp",
    url="https://github.com/nu-dialogue/antor",
    description = "Adaptive Natural Language Generation for Task-oriented Dialogue via Reinforcement Learning",
    license = "MIT", 
    python_requires='>=3.7',
    install_requires=requirements,
    include_package_data=True
)