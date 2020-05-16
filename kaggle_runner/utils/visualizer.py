from colorama import Fore, Style

import plotly.graph_objects as go
from googletrans import Translator

# ### Visualize model predictions
#
# Now, I will visualize the performance of the model on few validation samples.

# + {"_kg_hide-input": true}
translator = Translator()

def visualize_model_preds(model,val_data, x_valid, y_valid, indices=[0, 17, 1, 24]):
    comments = val_data.comment_text.loc[indices].values.tolist()
    preds = model.predict(x_valid[indices].reshape(len(indices), -1))

    for idx, i in enumerate(indices):
        if y_valid[i] == 0:
            label = "Non-toxic"
            color = f'{Fore.GREEN}'
            symbol = '\u2714'
        else:
            label = "Toxic"
            color = f'{Fore.RED}'
            symbol = '\u2716'

        print('{}{} {}'.format(color, str(idx+1) + ". " + label, symbol))
        print(f'{Style.RESET_ALL}')
        print("ORIGINAL")
        print(comments[idx].encode('utf-8')); print("")
        print("TRANSLATED")
        print(translator.translate(comments[idx].encode('utf-8')).text)
        # fig = go.Figure()

        # if list.index(sorted(preds[:, 0]), preds[idx][0]) > 1:
        #     yl = [preds[idx][0], 1 - preds[idx][0]]
        # else:
        #     yl = [1 - preds[idx][0], preds[idx][0]]
        # fig.add_trace(go.Bar(x=['Non-Toxic', 'Toxic'], y=yl,
        #     marker=dict(color=["seagreen", "indianred"])))
        # fig.update_traces(name=comments[idx])
        # fig.update_layout(xaxis_title="Labels", yaxis_title="Probability",
        #     template="plotly_white", title_text=("Predictions for validation "
        #     "comment #{}").format(idx+1))
        # fig.show()
