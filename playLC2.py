from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import my_vector


#model = OllamaLLM(model="llama3.2")
model = OllamaLLM(model="llama3.1:8b")


template = """
Du er en præcis dansk analyseassistent.

Du skal svare på brugerens spørgsmål udelukkende ud fra anmeldelserne nedenfor.

Regler:
- Svar på korrekt dansk.
- Opfind ikke oplysninger.
- Brug ikke mærkelige eller hjemmelavede ord.
- Hvis anmeldelserne ikke giver nok information, så sig det tydeligt.
- Underbyg svaret med konkrete eksempler fra anmeldelserne.
- Svar kort og struktureret.

Anmeldelser:
{reviews}

Spørgsmål:
{question}

Svar:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

chain.invoke

while True:
    print("\n\n-------------------------------")
    question = input("Stil dit spørgsmål (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = my_vector.retriever.invoke(question)
    for doc in reviews:
        print(doc.page_content[:500])
        print(doc.metadata)
        print("---")
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
