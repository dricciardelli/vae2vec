import matplotlib.pyplot as plt
import pickle
import gzip
import pylab

def load_file(filename, n_limit=None, skip=200):
	with open(filename, 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		p = u.load()
	if n_limit:
		p = p[skip:n_limit+skip]
	return p

def main():

	n_limit = 100
	skip = 300

	results_20 = list(zip(*enumerate(load_file("20_results.pkl", n_limit))))
	results_20_256 = list(zip(*enumerate(load_file("20_256_results.pkl", n_limit))))
	results_50_256 = list(zip(*enumerate(load_file("50_256_results.pkl", n_limit))))
	results_50 = list(zip(*enumerate(load_file("50_results.pkl", n_limit))))
	results_100 = list(zip(*enumerate(load_file("100_results.pkl", n_limit))))
	results_100_256 = list(zip(*enumerate(load_file("100_256_results.pkl", n_limit))))
	
	pylab.plot([x + skip for x in results_20[0]], results_20[1], label='Latent Dim: 20, LSTM Size: 128')
	pylab.plot([x + skip for x in results_20_256[0]], results_20_256[1], label='Latent Dim: 20, LSTM Size: 256')
	pylab.plot([x + skip for x in results_50_256[0]], results_50_256[1], label='Latent Dim: 50, LSTM Size: 256')
	pylab.plot([x + skip for x in results_50[0]], results_50[1], label='Latent Dim: 50, LSTM Size: 128')
	pylab.plot([x + skip for x in results_100[0]], results_100[1], label='Latent Dim: 100, LSTM Size: 128')
	pylab.plot([x + skip for x in results_100_256[0]], results_100_256[1], label='Latent Dim: 100, LSTM Size: 256')

	pylab.title("Cost per Epoch")
	pylab.xlabel("Epoch Number")
	pylab.ylabel("Cost")

	pylab.legend(loc='upper left')

	plt.show()



if __name__ == "__main__":
	main()