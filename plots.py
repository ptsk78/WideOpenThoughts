import matplotlib.pyplot as plt
import numpy as np

def makePlot(opt, optT, cor, corT, confMatr, correctProbs, incorrectProbs, times, confMatrT, correctProbsT, incorrectProbsT, timesT, im1, im2, fileName):
    fig = plt.figure(figsize=(30,15))
    gs = fig.add_gridspec(3,4)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('Confusion Matrix Train')
    ax.matshow(confMatr, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(len(confMatr)):
        for j in range(len(confMatr[0])):
            ax.text(x=j, y=i,s=confMatr[i, j], va='center', ha='center', size='small')
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title('Confusion Matrix Test')
    ax.matshow(confMatrT, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(len(confMatrT)):
        for j in range(len(confMatrT[0])):
            ax.text(x=j, y=i,s=confMatrT[i, j], va='center', ha='center', size='small')

    ax = fig.add_subplot(gs[1, 0])
    ax.set_title('Probs Correct Train')
    ax.set_xlim(-0.05,1.05)
    for i in range(len(correctProbs)):
        ax.hist(correctProbs[i], bins=len(correctProbs[i])//100, density=True, histtype='step')
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title('Probs Correct Test')
    ax.set_xlim(-0.05,1.05)
    for i in range(len(correctProbsT)):
        ax.hist(correctProbsT[i], bins=len(correctProbsT[i])//100, density=True, histtype='step')

    ax = fig.add_subplot(gs[2, 0])
    ax.set_title('Probs InCorrect Train')
    ax.set_xlim(-0.05,1.05)
    for i in range(len(correctProbs)):
        ax.hist(incorrectProbs[i], bins=len(incorrectProbs[i])//100, density=True, histtype='step')
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title('Probs InCorrect Test')
    ax.set_xlim(-0.05,1.05)
    for i in range(len(correctProbsT)):
        ax.hist(incorrectProbsT[i], bins=len(incorrectProbsT[i])//100, density=True, histtype='step', label=str(i))
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.set_title('Optimization')
    ax.plot(range(len(opt)), opt, label='train')
    ax.plot(range(len(optT)), optT, label='test')
    ax.legend()
    ax = fig.add_subplot(gs[0, 3])
    ax.set_title('Accuracy')
    ax.plot(range(len(cor)), cor)
    ax.plot(range(len(corT)), corT)

    ax = fig.add_subplot(gs[1, 2])
    ax.set_title('Optimization Second Half')
    ax.plot(range(len(opt)//2, len(opt)), opt[len(opt)//2:])
    ax.plot(range(len(optT)//2, len(optT)), optT[len(optT)//2:])
    ax = fig.add_subplot(gs[1, 3])
    ax.set_title('Accuracy Second Half')
    ax.plot(range(len(cor)//2, len(cor)), cor[len(cor)//2:])
    ax.plot(range(len(corT)//2, len(corT)), corT[len(corT)//2:])

    ax = fig.add_subplot(gs[2, 2])
    times = np.array(times)
    timesT = np.array(timesT)
    ax.set_title('Layer Times')
    ax.bar([i + 1.0 / 7.0 for i in range(times.shape[0])], times[:,0], width = 1.0/7.0)
    ax.bar([i + 2.0 / 7.0 for i in range(times.shape[0])], times[:,1], width = 1.0/7.0)
    ax.bar([i + 3.0 / 7.0 for i in range(times.shape[0])], times[:,2], width = 1.0/7.0)
    ax.bar([i + 4.0 / 7.0 for i in range(timesT.shape[0])], timesT[:,0], width = 1.0/7.0)
    ax.bar([i + 5.0 / 7.0 for i in range(timesT.shape[0])], timesT[:,1], width = 1.0/7.0)
    ax.bar([i + 6.0 / 7.0 for i in range(timesT.shape[0])], timesT[:,2], width = 1.0/7.0)

    ax = fig.add_subplot(gs[2, 3])

    im = np.zeros(((im1.shape[2] + 5)*5,(im1.shape[3]+5)*2,3), np.int32)
    for i in range(5):
        for j in range(im1.shape[2]):
            for k in range(im1.shape[3]):
                for l in range(3):
                    im[j + i * (im1.shape[3] + 5),k,l] = int(255.0 * im1[i,l,j,k])
        for j in range(im1.shape[2]):
            for k in range(im1.shape[3]):
                for l in range(3):
                    im[j + i * (im1.shape[3] + 5),k + im1.shape[3] + 5,l] = int(255.0 * im2[i,l,j,k])
    ax.imshow(im)

    plt.savefig(fileName)
    plt.close()
