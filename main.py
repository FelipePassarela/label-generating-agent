import pandas as pd
from crewai import Agent, Crew, Task
from langchain_openai import ChatOpenAI
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


def start_sentiment_analysis(entities: list, contents: list) -> list[str]:
    llm = ChatOpenAI(model="ollama/crewai-llama3.2:latest",
                     base_url="http://localhost:11434")
    
    # Uncomment the following lines to use the OpenAI API (recommended)
    # llm = LLM(model="gpt-4-turbo",
    #           api_key=os.environ["OPENAI_API_KEY"])

    labeler = Agent(role="Smart Data Labeler",
                    goal="Label the sentiments of texts.",
                    backstory="A Smart Data Labeler, responsible for reading and labeling the sentiments of texts.",
                    llm=llm)

    labeling_task = Task(
        description="Given the entity to which the text refer and its content, "
                    "label the sentiment about the entity in the text as 'Positive', 'Negative', 'Neutral' "
                    "or 'Irrelevant' (if the text does not relate to the entity)"
                    "The label MUST BE restricted to one of these four options.\n"
                    "Here is the text to be labeled:\n"
                    '"{text}"',
        expected_output="An one word phrase representing the sentiment of the text, "
                        "i.e., 'Positive', 'Negative', 'Neutral' or 'Irrelevant'.",
        agent=labeler)

    crew = Crew(agents=[labeler],
                tasks=[labeling_task],
                verbose=True)

    messages = [f"Entity: {entity}\nContent: {content}" for entity, content in zip(entities, contents)]
    inputs = [{"text": msg} for msg in messages]
    results = crew.kickoff_for_each(inputs=inputs)

    results = [res.raw for res in results]
    return results


def clean_preds(y_preds):
    for i, y in enumerate(y_preds):
        if "positive" in y.lower():
            y_preds[i] = "Positive"
        elif "negative" in y.lower():
            y_preds[i] = "Negative"
        elif "neutral" in y.lower():
            y_preds[i] = "Neutral"
        elif "irrelevant" in y.lower():
            y_preds[i] = "Irrelevant"
        else:
            y_preds[i] = "Unclassified"


if __name__ == "__main__":
    df = pd.read_csv('data/processed/twitter_validation.csv')
    df = df.sample(100, random_state=42)

    entities = df['entity'].tolist()
    texts = df['text'].tolist()
    y_true = df['sentiment'].tolist()

    y_preds = start_sentiment_analysis(entities, texts)
    clean_preds(y_preds)

    print("Accuracy:", accuracy_score(y_true, y_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_preds))
    print("Classification Report:\n", classification_report(y_true, y_preds, zero_division=0))

    df_preds = pd.DataFrame({
        'entity': entities, 
        'text': texts, 
        'sentiment': y_true, 
        'predicted_sentiment': y_preds})
    df_preds.to_csv('preds.csv', index=False)