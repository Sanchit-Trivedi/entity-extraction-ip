from nltk.stem import WordNetLemmatizer, PorterStemmer
from spacy.matcher import PhraseMatcher
from nltk.corpus import stopwords
from scipy import spatial
import gensim.models.keyedvectors as word2vec
import html.parser as HTMLParser
import pandas as pd
import numpy as np
import wordcloud
import pickle
import spacy
import nltk
import time
import csv
import re
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 

# LOADING SKILLS IN MATCHER
start = time.time()
skills = pd.read_csv('skills.txt', sep='\n', header=None)
print ("Loading Spacy")
nlp = spacy.load("en_core_web_sm")
print("Making skill words")
skill_words = [nlp(text) for text in skills[0].dropna(axis=0)]
print("Done")
matcher = PhraseMatcher(nlp.vocab)
matcher.add('Skills', None, *skill_words)
print("Skills loaded in matcher")
print ("Total time taken to load skills : ",time.time()-start)

# LOADING WORD2VEC MODEL
print ()
print ("Loading Skill2vec Model")
start = time.time()
model = word2vec.KeyedVectors.load_word2vec_format('duyet_word2vec_skill.bin', binary=True)
print ("Model loaded in : ",time.time()-start)

# HTML PARSER FOR SKILL TRANSFORM
print ()
print("HTML Parser for Skill Transform")
start = time.time()
html_parser = HTMLParser.HTMLParser()
wordnet_lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
print ("Skill transform all set in time : ",time.time()-start)

def get_skills(t2):
	doc = nlp(t2)
	matches = matcher(doc)
	done = set()
	for match_id, start, end in matches:
		rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
		span = doc[start: end]  # get the matched slice of the doc
		if span.text not in done:
			done.add(span.text)
	return ",".join(map(str,list(done)))

def avg_feature_vector(words, model, num_features):
		#function to average all words vectors in a given paragraph
		featureVec = np.zeros((num_features,), dtype="float32")
		nwords = 0
		words = [skill_transform(word) for word in words]
		#list containing names of words in the vocabulary
		index2word_set = set(model.index2word) # this is moved as input param for performance reasons
		for word in words:
			if word in index2word_set:
				nwords = nwords+1
				featureVec = np.add(featureVec, model[word])
		if(nwords>0):
			featureVec = np.divide(featureVec, nwords)
		return featureVec

def compare_two_list_skills(skills_1, skills_2):
	sentence_1_avg_vector = avg_feature_vector(skills_1.split(), model=model, num_features=300)
	sentence_2_avg_vector = avg_feature_vector(skills_2.split(), model=model, num_features=300)
	sen1_sen2_similarity =  1 - spatial.distance.cosine(sentence_1_avg_vector,sentence_2_avg_vector)
	return sen1_sen2_similarity


def skill_transform(skill, remove_stopwords = True):
	skill = str(skill)
	skill = html_parser.unescape(skill)	
	skill = skill.replace("_", " ").split()
	skill = " ".join([sk for sk in skill if sk])
	skill = re.sub(r"\(.*\)", "", skill)
	skill = skill.replace("-", "") \
		.replace(".", "") \
		.replace(",", "") \
		.replace("-", "") \
		.replace(":", "") \
		.replace("(", "") \
		.replace(")", "") \
		.replace(u"åá", "") \
		.replace(u"&", "and") \
		.replace(" js", "js") \
		.replace("-js", "js") \
		.replace("_js", "js") \
		.replace("java script", "js") 
	skill = skill.lower()
	# Special cases replace
	special_case = {}
	special_case["Computer"] = [""]
	special_case["javascript"] = [ "js", "java script", "javascripts", "java scrip" ]
	special_case["wireframe"] = [ "wireframes", "wire frame", "wire frames", "wire-frame", "wirefram", "wire fram", "wireframing" ]
	special_case["OOP"] = [  "object oriented", "object oriented programming", ]
	special_case["OOD"] = [ "object oriented design", ]
	special_case["OLAP"] = [ "online analytical processing",  ]
	special_case["Ecommerce"] = [ "e commerce",  ]
	special_case["consultant"] = [ "consulting",  ]
	special_case["ux"] = [ "user experience", "web user experience design", "user experience design", "ux designer", "user experience/ux" ]
	special_case["html5"] = [ "html 5",  ]
	special_case["j2ee"] = [ "jee",  ]
	special_case["osx"] = [ "mac os x", "os x" ]
	special_case["senior"] = [ "sr" ]
	special_case["qa"] = [ "quality",  ]
	special_case["bigdata"] = [ "big data",  ]
	special_case["webservice"] = [ "webservices", "website", "webapps" ]
	special_case["xml"] = [ "xml file", "xml schemas", "xml/json", "xml web service" ]
	special_case["bigdata"] = [ "big data",  ]
	special_case["nlp"] = [ "natural language process", "natural language", "nltk" ]
	# Skills we added
	special_case["ml"] = ["machine learning", "machinelearning", "machine_learning", "machine_learn"]
	special_case["dbms"] = ["database"]
	for root_skill in special_case:
		if skill in special_case[root_skill]:
			skill = root_skill
	# Special case regex
	special_case_regex = {
		r'^angular.*$': 'angularjs',
		r'^node.*$': 'nodejs',
		r'^(.*)[_\s]js$': '\\1js',
		r'^(.*) js$': '\\1js',
	}
	for regex_rule in special_case_regex:
		after_skill = re.sub(regex_rule, special_case_regex[regex_rule], skill)
		if after_skill != skill:
			skill = after_skill
			break
	if len(skill) > 2:
		skill_after = skill.split(" ")
		skill_after = [wordnet_lemmatizer.lemmatize(sk, pos="v") for sk in skill_after]
		skill_after = " ".join(skill_after)
		skill = skill_after
	# skill stopwords 
	if remove_stopwords:
		skill_stopwords = [ "app", "touch", "the", "application" ]
		skill_after = skill.split(" ")
		skill = " ".join([ sk for sk in skill_after if sk not in skill_stopwords ])
	skill = skill.lower().strip().replace(" ", "_")
	skill = re.sub(' +',' ', skill)
	# NOTE: replace js tail
	skill = re.sub('js$','', skill)
	return skill

def remove_stopwords(doc):
	words_list = []
	for token in doc:
		if token.is_stop:
			pass
		else:
			words_list.append(token.text)
	return words_list

def extract_softskills(jd):
	ps = " ".join([i[0] for i in model.similar_by_word("problem_solving")[0:5]])
	tw = " ".join([i[0] for i in model.similar_by_word("teamwork")[0:5]])
	ls = " ".join([i[0] for i in model.similar_by_word("leadership")[0:5]])
	ps += " problem_solving"
	tw += " teamwork team_work"
	ls += " leadership"
	jd = remove_stopwords(nlp(jd))
	softskill = []
	jd = " ".join(jd)
	search = [ps, tw, ls]
	for word in search:
		if word==ps:
			if compare_two_list_skills(jd, word)>0.87:
				softskill.append("problem_solving")
		elif word==tw:
			if compare_two_list_skills(jd, word)>0.92:
				softskill.append("teamwork")
		else:
			if compare_two_list_skills(jd, word)>0.89:
				softskill.append("leadership")
	return softskill

def getGap(job_skills, resume_skills, threshold = 0.65):
	resume_skills = [skill_transform(x) for x in resume_skills]
	skills_not_found = []
	for req_skill in job_skills:
		skill_found = 0
		for given_skill in resume_skills:
			score = compare_two_list_skills(req_skill, given_skill)
			if score >= threshold:
				skill_found = 1
				break
		if not skill_found:
			skills_not_found.append(req_skill)
	return skills_not_found

def find_skill_gap(job_description, resume_skills):
	soft_skills = extract_softskills(job_description)
	job_skills = get_skills(job_description).split(",")
	skill_gap = getGap(job_skills, resume_skills)
	return job_skills, soft_skills, skill_gap

job_description = "As a member of our Software Engineering Group, we look first and foremost for people who are passionate around solving business problems through innovation and engineering practices. You'll be required to apply your depth of knowledge and expertise to all aspects of the software development lifecycle, as well as partner continuously with your many stakeholders on a daily basis to stay focused on common goals. We embrace a culture of experimentation and constantly strive for improvement and learning. You'll work in a collaborative, trusting, thought-provoking environment-one that encourages diversity of thought and creative solutions that are in the best interests of our customers globally. This role requires a wide variety of strengths and capabilities, including: BS/BA degree or equivalent experience Strong understanding of Python, Spark, Scala, Java, Microservice design patterns, Object Oriented, Functional & Reactive Programming Hands-on experience with Pytest, ScalaTest, and experience using Jenkins, SonarQube, GIT, Maven. Apache Kafka a plus, proficiency in SQL, preferably across multiple databases - Cassandra or any NoSQL database is a must. Advanced knowledge of application, data, and infrastructure architecture disciplines Understanding of architecture and design across all systems Working proficiency in developmental toolsets Knowledge of industry-wide technology trends and best practices Ability to work in large, collaborative teams to achieve organizational goals Passionate about building an innovative culture Proficiency in one or more modern programming languages Understanding of software skills such as business analysis, development, maintenance, and software improvement JPMorgan Chase & Co., one of the oldest financial institutions, offers innovative financial solutions to millions of consumers, small businesses and many of the world's most prominent corporate, institutional and government clients under the J.P. Morgan and Chase brands. Our history spans over 200 years and today we are a leader in investment banking, consumer and small business banking, commercial banking, financial transaction processing and asset management. We recognize that our people are our strength and the diverse talents they bring to our global workforce are directly linked to our success. We are an equal opportunity employer and place a high value on diversity and inclusion at our company. We do not discriminate on the basis of any protected attribute, including race, religion, color, national origin, gender, sexual orientation, gender identity, gender expression, age, marital or veteran status, pregnancy or disability, or any other basis protected under applicable law. In accordance with applicable law, we make reasonable accommodations for applicants' and employees' religious practices and beliefs, as well as any mental health or physical disability needs."
resume_skills = ["jdbc", "html", "python", "sql", "oracle"]
skill_gap = find_skill_gap(job_description, resume_skills)
print ("Following are the JOB skills : ")
print (skill_gap[0])
print ()
print ("Following are the required soft skills : ")
print (skill_gap[1])
print ()
print ("Following are the skills you have : ")
print (resume_skills)
print ()
print ("Following are the required skills you don't have : ")
print (skill_gap[2])




