"""
Command-line interface defining the available tools in the SLM Toolkit.
This module registers click commands for text summarization, generation,
translation, RAG querying, MCQ generation, and notes generation.
"""
import click

from services.summarizer import TextSummarizer
from services.generator import TextGenerator
from services.translator import Translator
from services.rag import RAGPipeline
from services.mcq_generator import MCQGenerator
from services.notes_generator import NotesGenerator


@click.group()
def cli():
    """
    SLM Tool CLI base command group.
    """
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


@cli.command()
@click.argument("topic")
@click.argument("count", type=int)
@click.argument("category", type=click.Choice(["easy", "medium", "difficult"], case_sensitive=False))
def questions(topic, count, category):

    generator = MCQGenerator()

    print(f"\nGenerating {count} {category} questions about '{topic}'...")
    print(generator.generate_questions(topic, count, category))


@cli.command()
@click.argument("topic")
def notes(topic):

    generator = NotesGenerator()

    print(f"\nGenerating student-friendly notes about '{topic}'...\n")
    print(generator.generate_notes(topic))
