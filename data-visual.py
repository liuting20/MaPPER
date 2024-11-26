import matplotlib.pyplot as plt


x = list(map(lambda v: v * 15 if v < 100 else 160, [196, 4.09, 3.25, 4.09, 4.09, 2.98, 2.77]))
y = [85.31, 83.23, 82.43, 81.71, 82.35, 81.52, 86.38]
label = ['Fully fine-tuning', 'Adapter', 'LoRA', 'AdaptFormer', 'CM Adapter', 'MRS-Adapter', 'MaPPER (Ours)']
# markers = ['o', '^', 'v', 'P', 'd', '3', 's']
markers = ['o']*6 + ['*']
xttext = [(-88, -3), (10, -3), (-33, -3), (10, -3), (10, -6), (-73, -3), (10, -3)]
plt.xticks([0,  30, 60, 90, 150, 200 ], labels=['0', '2', '4', '6', '195', '200'])
plt.yticks([81, 82, 83, 84, 85, 86, 87], labels=['81', '82', '83', '84', '85', '86', '87'])

plt.xlim([0, 200])
plt.ylim([81, 87])


if __name__ == '__main__':
   
    for i in range(len(x)):
        plt.scatter(x[i], y[i], marker=markers[i], s=100)
        plt.annotate(label[i], (x[i], y[i]), textcoords="offset points", xytext=xttext[i], ha='left')


  
    plt.title('Comparision to Others PETL Methods')
    plt.xlabel('Tunable backbone parameters (M)')
    plt.ylabel('RefCOCO Val')
    plt.savefig('./aaa.pdf', format='pdf', bbox_inches='tight', )

 
    plt.show()
