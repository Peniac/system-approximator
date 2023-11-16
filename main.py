from system import System


if __name__ == "__main__":
	system = System()

	system.run(iters=1000)
	system.plot_data()