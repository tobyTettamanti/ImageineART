import cohere
import hub
import json


def image_label_test():
    ds = hub.load('hub://activeloop/flickr30k')
    ds.tensors.keys()  # dict_keys(['images', 'labels'])
    file = open("flickr30k_nohumans_desc","w")
    for i in range(31782):
        desc = str(ds.texts[i].data())  # Fetch the text description
        desc = desc[12:len(desc)-2] #parse quotes
        desc = desc.lower()
        if check_no_humans(desc):
            desc += "\n"
            file.write(desc)
            print(desc)
            file.write("--SEPARATOR--\n")
    file.close()
    print("File successfully written")

def check_no_humans(desc):
    humans = ["man", "men", "person", "people", "child", "kid",
   "boy", "girl", "male", "baby", "guy", "adult",  "er", "ist", "group", "bride",
    "lady", "ladies", "dog", "individual", "someone", "mate", "student", "team"]

    for word in humans:
        if word in desc:
            return False
    return True

def add_manual_prompts():
    wrFile = open("manual_edits", "w+")
    rdFile = open("manual_prompts.txt", "r")
    for i in range(176):
            line = rdFile.readline()
            wrFile.write(line)
            wrFile.write("--SEPARATOR--\n")
    wrFile.close()
    rdFile.close()

def string_to_json(prompts):
    import json

    promptArr = prompt.split("\n")
    # json string data
    prompts = '{"prompts":promptArr}'
    # convert string to  object
    json_object = json.loads(prompts)

    # check new data type
    print(type(json_object))


def cohere_test():
    co = cohere.Client('k1qhkmPpmT7w4ryiEk5hJvjiBlwyieie7ogGSB0i')
    response = co.generate(
        model='large',
        prompt='',
        max_tokens=50,
        temperature=1,
        num_generations=5,
        k=0,
        p=0.75)
    for i in range(5):
        print('Prediction: {}'.format(response.generations[i].text))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    string_to_json("k1", "v1", "k2", "v2")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
