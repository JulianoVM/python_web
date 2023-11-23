# Atividade Flask + Projeto final Flask
# ADENDO!!!!!!!!!!!!
cada arquivo clf_{nome do classificador}.py tem um "plt.savefig()" para salvar a imagem em um caminho desejado, para o programa funcionar e não reclamar que o caminho não existe, dentro da pasta "machine_learnin_proj" tem uma pasta "images", clique com o botão direito e copie o caminho da pasta, cole esse caminho dentro da "plt.savefig()" de cada arquivo "cls_{nome do classificador}.py" e coloque o nome desejável do png.
# Exemplo:
plt.savefig('c:/User/Downloads/python_web/machine_learning_proj/images/{nome_da_imagem}.png')
# CUIDADO!!!!
se estiver no windows, ele copia o caminho de maneira diferente do shell do Linux, então se rodar o programa por um terminal Shell ou PowerShell, certifique de mudar a "\" que separa os caminhos
# Exemplo 2:
windows: 'C:\User\Alguma_coisa\bla\bla'
Linus (PowerShell ou Shell): 'C:/User/Alguma_coisa/bla/bla'
(as divisões do arquivo são diferentes, "\" em um e "/" no outro)
