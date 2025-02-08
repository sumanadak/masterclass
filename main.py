import cohere
co = cohere.Client('') # This is your trial API key

response = co.embed(
  model='embed-english-v3.0',
  texts=["apple"],
  input_type='classification',
  truncate='NONE'
)

print('Embeddings: {}'.format(response.embeddings))
