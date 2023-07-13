import matplotlib.pyplot as plt


def plot_clusters(cluster_label, firms, firm_names, data_array):
    if cluster_label == -1:
        print(f'Noise: {firms}')
        title = 'Noise'
    else:
        print(f'Cluster {cluster_label + 1}: {firms}')
        title = f'Cluster {cluster_label + 1}'

    # Plot the line graph for firms in the cluster
    for firm in firms:
        firm_index = list(firm_names).index(firm)
        firm_data = data_array[firm_index]

        plt.plot(range(1, len(firm_data) + 1), firm_data, label=firm)

    plt.xlabel('Characteristics')
    plt.ylabel('Data Value')
    plt.title(title)

    # List the firm names on the side of the graph
    if len(firms) <= 10:
        plt.legend(loc='center right')
    else:
        plt.legend(loc='center right', title=f'Total Firms: {len(firms)}', labels=firms[:10] + ['...'])

    plt.show()