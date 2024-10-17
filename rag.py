def extract_course_code(text) -> list[str]:
    pattern = r'\b(?:geds?|stats?|maths?|cosc|seng|itgy)\s*\d{3}\b'
    match = re.findall(pattern, text, re.IGNORECASE)
    return match if match else None

def get_docs(query: str, top_k: int) -> list[str]:
    course_code = extract_course_code(query)
    exact_matches = []

    if course_code:
        course_code = [code.lower() for code in course_code]
        
        exact_matches = [
            x['content'] for x in data['metadata']
            if any(code in x['content'].lower() for code in course_code)
        ]
    
    remaining_slots = top_k - len(exact_matches)
    
    if remaining_slots > 0:
        xq = encoder.encode(query)
        res = index.query(vector=xq.tolist(), top_k=remaining_slots if exact_matches else top_k, include_metadata=True)
        
        embedding_matches = [x["metadata"]['content'] for x in res["matches"]]
        
        exact_matches.extend(embedding_matches)
    
    return exact_matches[:top_k]

def get_response(query: str, docs: list[str]) -> str:
    system_message = (
        "You are Anjibot, the AI course rep of 400 Level Computer Science department. You are always helpful, jovial, can be sarcastic but still sweet.\n"
        "Provide the answer to class-related queries using\n"
        "context provided below.\n"
        "If you don't the answer to the user's question based on your pretrained knowledge and the context provided, just direct the user to Anji the human course rep.\n"
        "Anji's phone number: 08145170886.\n\n"
        "CONTEXT:\n"
        "\n---\n".join(docs)
        )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return chat_response.choices[0].message.content

def handle_query(user_query: str):

    docs = get_docs(user_query, top_k=5)

    response = get_response(user_query, docs=docs)

    for word in response.split():
            yield word + " "
            time.sleep(0.05)
