# ontology-as-a-service

Deliverables
1) Upload code on GitHub
2) Code for API (e.g. Flask)


Ongoing improvements
1) Filter rows with both label and comment being empty
2) Remove underscores (_) and split words connected (e.g. placeOfBirth --> place of birth)
3) Focus on the labels, and create a Faiss index using the vec_labels
4) Use the query embedding to search on the index the top-k related properties/classes
5) Use template from EPSO UI (Dash-based)
6) Deploy UI to GRNet instance or ask IM for 

Future work
1) Check also the comments
2) Which vocabularies to consider as raw data
3) Export results to csv/Excel


Later
1) Ability to load vocabularies (the user to provide a new ontology that will be inserted in the database for indexing)