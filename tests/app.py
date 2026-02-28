from slidedeckai.core import SlideDeckAI


slide_generator = SlideDeckAI(
    model='[gg]gemini-2.5-flash',
    topic='Make a slide deck on AI',
    api_key='AIzaSyA93DqRPLmuovHDk2KU3Y9R9SaHOEXVPRU',  # Or set via environment variable
)
pptx_path = slide_generator.generate()
print(f'🤖 Generated slide deck: {pptx_path}')