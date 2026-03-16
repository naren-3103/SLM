import click

from services.summarizer import TextSummarizer
from services.generator import TextGenerator
from services.translator import Translator
from services.rag import RAGPipeline


@click.group()
def cli():
    pass


@cli.command()
@click.argument("text")
def summarize(text):

    s = TextSummarizer()

    print(s.summarize_text(text))


@cli.command()
@click.argument("prompt")
def generate(prompt):

    g = TextGenerator()

    print(g.generate(prompt))


@cli.command()
@click.argument("text")
@click.argument("lang")
def translate(text, lang):

    t = Translator()

    print(t.translate(text, lang))


@cli.command()
@click.argument("question")
def ask(question):

    rag = RAGPipeline()

    answer = rag.ask(question)

    print("\nAnswer:\n")

    print(answer)