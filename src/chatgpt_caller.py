import numpy as np
import openai
import json
import hashlib
import os.path
import pandas as pd
from sklearn.cluster import KMeans
from rich.progress import track

class chatgpt_caller():
    def __init__(self, folder, api_key):
        self.api_key = api_key
        
        openai.api_key = self.api_key

        self.COMPLETIONS_MODEL = "gpt-3.5-turbo"
        self.EMBEDDING_MODEL = "text-embedding-ada-002"
        self.CONTEXT_TOKEN_LIMIT = 1500
        self.TOKENS_PER_TOPIC = 2000
        self.TOPIC_NUM_MIN = 3
        self.TOPIC_NUM_MAX = 10

        self.content = ""
        self.embeddings = []
        self.sources = []

        self.folder = folder
        self.create_folder(folder)

    def create_folder(self, folder_path):
        # create path if not exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def get_embedding(self, text):
        embed_folder = os.path.join(self.folder, 'embeddings/cache/')
        self.create_folder(embed_folder)
        tmpfile = embed_folder+hashlib.md5(text.encode('utf-8')).hexdigest()+".json"
        if os.path.isfile(tmpfile):
            with open(tmpfile , 'r', encoding='UTF-8') as f:
                return json.load(f)
    
        result = openai.Embedding.create(
            model=self.EMBEDDING_MODEL,
            input=text,
            api_key=self.api_key
        )

        with open(tmpfile, 'w',encoding='utf-8') as handle2:
            json.dump(result["data"][0]["embedding"], handle2, ensure_ascii=False, indent=4) # type: ignore
        
        return result["data"][0]["embedding"] # type: ignore
        
    def get_topic_num(self):
        num = int(len("".join(self.sources))/self.TOKENS_PER_TOPIC)
        if num < self.TOPIC_NUM_MIN:
            return self.TOPIC_NUM_MIN
        if num > self.TOPIC_NUM_MAX: 
            return self.TOPIC_NUM_MAX
        return num
        
    def get3questions(self):
        matrix = np.vstack(self.embeddings)
        print(matrix.shape)
    
        df = pd.DataFrame({"embedding":self.embeddings,"p":self.sources})
        n_clusters = self.get_topic_num()
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        kmeans.fit(matrix)
        df["Cluster"] = kmeans.labels_  

        df2 = {"tokens":[], "prompts":[]}
        for i in range(n_clusters):
            ps = df[df.Cluster == i]["p"].values
            ctx = "\n".join(ps)[:self.CONTEXT_TOKEN_LIMIT]
            prompt = f"Suggest a simple, clear, single, short question base on the context, answer in the same language of context\n\nContext:\n{ctx}\n\nAnswer with the language used in context, the question is:"
            df2["tokens"].append(len(" ".join(ps)))
            df2["prompts"].append(prompt)
        df2 = pd.DataFrame(df2)
        
        print("######questions#######")
        questions = []
        for prompt in df2.prompts.sample(3).values:
            print(prompt)
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content":prompt}], api_key=self.api_key)
                question = completion.choices[0].message.content # type: ignore
                questions.append(question)
                print(question)
            except Exception as e:
                print("Error when deal with questions: ", e)
            print("***********************")
        return questions

    def file2embedding(self, text):
        self.content = text

        self.sources = self.content.split('\n')
        temp_sources = []
        for source in track(self.sources):
            if source.strip() == '':
                continue
            embed = self.get_embedding(source)
            if embed is not None:
                self.embeddings.append(embed)
                temp_sources.append(source)
        self.sources = temp_sources

        self.questions = self.get3questions()
    
        with open(os.path.join(self.folder, "embed_result.json"), 'w',encoding='utf-8') as f:
            json.dump({"sources": self.sources, "embeddings": self.embeddings, "questions": self.questions}, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    folder = "./"
    text_file = "text.txt"
    chatgpt_caller(folder).file2embedding(text_file)