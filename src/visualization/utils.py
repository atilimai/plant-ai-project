import matplotlib.pyplot as plt
import seaborn as sns

def set_project_plot_style():

    plt.style.use('ggplot')

    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    plt.rcParams['figure.dpi'] = 300

    sns.set_palette("viridis")

    print("Proje grafik stili başarıyla uygulandı.")

def clean_label(label):
    return label.replace("_", " ").replace("-", " ").title()
