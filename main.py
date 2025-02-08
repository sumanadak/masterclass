import cohere
co = cohere.Client('xz6VhvXnfldENCPxSIWa19qEumptDOjH2tPoXA1F') # This is your trial API key

response = co.embed(
  model='embed-english-v3.0',
  texts=["When are you open?", "When do you close?", "What are the hours?", "Are you open on weekends?", "Are you available on holidays?", "How much is a burger?", "What\'s the price of a meal?", "How much for a few burgers?", "Do you have a vegan option?", "Do you have vegetarian?", "Do you serve non-meat alternatives?", "Do you have milkshakes?", "Milkshake", "Do you have desert?", "Can I bring my child?", "Are you kid friendly?", "Do you have booster seats?", "Do you do delivery?", "Is there takeout?", "Do you deliver?", "Can I have it delivered?", "Can you bring it to me?", "Do you have space for a party?", "Can you accommodate large groups?", "Can I book a party here?"],
  input_type='classification',
  truncate='NONE'
)

print('Embeddings: {}'.format(response.embeddings))
