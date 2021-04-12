class GDParams():

    def __init__(self, n_batch=100, eta=0.001, n_epochs=40, lam=0, eta_min=1e-5, eta_max=1e-1, n_s=500, plot=True, plot_dir='', cyclical_eta=True):
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epochs
        self.lam = lam
        self.plot = plot
        self.plot_dir = plot_dir
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.n_s = n_s
        self.cyclical_eta = cyclical_eta
        if cyclical_eta:
            if plot_dir != '':
                self.plot_dir += '_'
            self.plot_dir += '/cyclical_eta'

    def next_cyclical_learning_rate(self, t, l):
        if t <= (2*l + 1) * self.n_s and t >= 2*l*self.n_s:
            return self.eta_min + (t - 2*l*self.n_s) * (self.eta_max - self.eta_min) / self.n_s 
        return self.eta_max - (t - (2*l + 1) * self.n_s) * (self.eta_max - self.eta_min) / self.n_s 

    def __str__(self):
        return f'{self.n_epochs}_epochs_{self.lam}_lambda'
