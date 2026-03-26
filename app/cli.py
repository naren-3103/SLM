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
@click.option("--rebuild", is_flag=True, default=False, help="Force rebuild of the vector index.")
def ask(question, rebuild):

    rag = RAGPipeline(force_rebuild=rebuild)

    result = rag.ask(question)

    print("\nAnswer:\n")
    print(result["answer"])

    if result.get("sources"):
        print("\nSources:")
        for i, src in enumerate(result["sources"], start=1):
            print(f"  [{i}] {src['source']} — page {src['page']}  (score: {src['score']:.3f})")