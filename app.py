import streamlit as st
import tensorflow as tf
import spacy
from spacy.lang.en import English
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(5, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made in ",
        image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(25), height=px(25)),
        " with ❤️ by ",
        link("https://www.linkedin.com/in/kishanmishra1/", "@Kishanmishra1"),
        br()
    ]
    layout(*myargs)








# from utils import spacy_function, make_predictions, example_input
example_input = '''
Hepatitis C virus (HCV) and alcoholic liver disease (ALD), either alone or in combination, count for more than two thirds of all liver diseases in the Western world. 
There is no safe level of drinking in HCV-infected patients and the most effective goal for these patients is total abstinence. Baclofen, a GABA(B) receptor agonist, represents a promising pharmacotherapy for alcohol dependence (AD). 
Previously, we performed a randomized clinical trial (RCT), which demonstrated the safety and efficacy of baclofen in patients affected by AD and cirrhosis. 
The goal of this post-hoc analysis was to explore baclofen's effect in a subgroup of alcohol-dependent HCV-infected cirrhotic patients. 
Any patient with HCV infection was selected for this analysis. Among the 84 subjects randomized in the main trial, 24 alcohol-dependent cirrhotic patients had a HCV infection; 12 received baclofen 10mg t.i.d. and 12 received placebo for 12-weeks. 
With respect to the placebo group (3/12, 25.0%), a significantly higher number of patients who achieved and maintained total alcohol abstinence was found in the baclofen group (10/12, 83.3%; p=0.0123). Furthermore, in the baclofen group, compared to placebo, there was a significantly higher increase in albumin values from baseline (p=0.0132) and a trend toward a significant reduction in INR levels from baseline (p=0.0716). 
In conclusion, baclofen was safe and significantly more effective than placebo in promoting alcohol abstinence, and improving some Liver Function Tests (LFTs) (i.e. albumin, INR) in alcohol-dependent HCV-infected cirrhotic patients. Baclofen may represent a clinically relevant alcohol pharmacotherapy for these patients.
'''




def model_prediction(abstract):

    objective = ''
    background = ''
    method = ''
    conclusion = ''
    result = ''
    def split_chars(text):
        return " ".join(list(text))
    loaded_model=tf.keras.models.load_model('skimlit_tribrid_model')
    nlp = English() # setup English sentence parser
    sentencizer = nlp.create_pipe("sentencizer") # create sentence splitting pipeline object
    nlp.add_pipe('sentencizer')
    doc = nlp(abstract) 
    abstract_lines = [str(sent) for sent in list(doc.sents)] # return detected sentences from doc in string type (not spaCy token type)
    total_lines_in_sample = len(abstract_lines)
    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15) 
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]
    test_abstract_pred_probs = loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                   test_abstract_total_lines_one_hot,
                                                   tf.constant(abstract_lines),
                                                   tf.constant(abstract_chars)))
    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    classes=['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    test_abstract_pred_classes = [classes[i] for i in test_abstract_preds]
   
    for i, line in enumerate(abstract_lines):
    
        if test_abstract_pred_classes[i] == 'OBJECTIVE':
            objective = objective + line
        
        elif test_abstract_pred_classes[i] == 'BACKGROUND':
            background = background + line
        
        elif test_abstract_pred_classes[i] == 'METHODS':
            method = method + line
        
        elif test_abstract_pred_classes[i] == 'RESULTS':
            result = result + line
        
        elif test_abstract_pred_classes[i] == 'CONCLUSIONS':
            conclusion = conclusion + line

    return objective, background, method, conclusion, result

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
"""



def main():
    
    st.set_page_config(
        page_title="PredMed",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title('PredMED📄')
    footer()
    st.markdown(reduce_header_height_style, unsafe_allow_html=True)
    st.subheader('NLP model to classify medical abstracts into descriptive reading sections.\n')
    st.subheader("Replicated Paper :[Paper Link](https://arxiv.org/abs/1710.06071)")
    # creating model, tokenizer and labelEncoder
    
    col1, col2 = st.columns(2)

    with col1:
        st.write('#### Entre Abstract Here !!')
        abstract = st.text_area(label='', height=None)
        # model = st.selectbox('Choose Model', ('Simple Model -> 82%', "Beart Model -> 89%"))

        agree = st.checkbox('Show Example Abstract')
        if agree:
            st.info(example_input)

        predict = st.button('Extract !')
    
    # make prediction button logic
    if predict:
        with st.spinner('Wait for prediction....'):
            objective, background, methods, conclusion, result = model_prediction(abstract)
        with col2:
            if len(objective)!=0:
                st.markdown(f'### Objective : ')
                st.write(f'{objective}')
            if len(background)!=0:
                st.markdown(f'### Background : ')
                st.write(f'{background}')
            if len(methods)!=0:
                st.markdown(f'### Methods : ')
                st.write(f'{methods}')
            if len(result)!=0:
                st.markdown(f'### Result : ')
                st.write(f'{result}')
            if len(conclusion)!=0:
                st.markdown(f'### Conclusion : ')
                st.write(f'{conclusion}')



if __name__=='__main__': 
    main()
    
