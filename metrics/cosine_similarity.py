from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import requests
import base64
from scipy.spatial.distance import cosine

testing_jsonlines=open("testing_card_json.jl","r").readlines()
path='/home/keerthanamanoharan/Documents/office_work/Pic2Code/gitlab/mystique/metrics/testing_data/'
def build_generated_card_json(images):
	generated_jsonlines_file=open("generated_card_json.jl","r")
	generated_images=[]
	for josnline in generated_jsonlines_file:
		js=json.loads(josnline)
		filename=js.get('filename','')
		generated_images.append(filename)

	generated_jsonlines_file=open("generated_card_json.jl","a")
	for image in images:
		print(path+image)
		if image not in generated_images:
			
			base64_string=''
			with open(path+image, "rb") as image_file:
				base64_string = base64.b64encode(image_file.read()).decode()
			response=requests.post('http://0.0.0.0:5050/predict_json',data=json.dumps({"image":base64_string}),headers={"Content-Type":"application/json"})
			content={"filename":str(image)}
			content["card_json"]=response.json().get('card_json')
			generated_jsonlines_file.write(json.dumps(content))
			generated_jsonlines_file.write("\n")







def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)[0][1]
    
def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


images=[]
for josnline in testing_jsonlines:
	js=json.loads(josnline)
	filename=js.get('filename','')
	images.append(filename)

build_generated_card_json(images)
generated_jsonlines=open("generated_card_json.jl","r").readlines()

	
cosine_similarities={}.fromkeys(images,'')
jaccard_similarities={}.fromkeys(images,'')
pattern=re.compile(r"\"\s*url\s*\"\s*\:\s*\"[^\"]*\"",re.IGNORECASE)
for line in testing_jsonlines:
	js=json.loads(line)
	test_card_json=js.get('card_json','')
	filename=js.get('filename','')
	generated_card_json=''
	for gen_jsonline in generated_jsonlines:
		if json.loads(gen_jsonline).get('filename','')==filename:
			generated_card_json=json.loads(gen_jsonline).get('card_json','')
			break
	test_card_json=re.sub(pattern,'"url": ""',json.dumps(test_card_json))
	generated_card_json=re.sub(pattern,'"url": ""',json.dumps(generated_card_json))
	cosine_similarities[filename]=str(get_cosine_sim(test_card_json,generated_card_json))
	jaccard_similarities[filename]=str(get_jaccard_sim(test_card_json,generated_card_json))
	
print("Cosiine Similarities:\n",json.dumps(cosine_similarities,indent=2))
print("Average Cosine Similarity:",round( (sum([float(l) for l in list(cosine_similarities.values())])/len(list(cosine_similarities.values())))*100   ))
print("Jaccard Similarities:\n",json.dumps(jaccard_similarities,indent=2))
print("Average Jaccard Similarity:",round(  (sum([float(l) for l in list(jaccard_similarities.values())])/len(list(jaccard_similarities.values())))*100   ))
