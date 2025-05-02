import torch
import models.surrogates.bayesian.deepbnn.pyro as pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Define a BFN for regression
class BayesianFlowNetwork:
    def __init__(self, input_dim=1, output_dim=1, num_steps=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        
        # Prior parameters (mean and variance of weights)
        self.prior_mean = torch.zeros(input_dim, output_dim)
        self.prior_var = torch.ones(input_dim, output_dim)
        
        # Learned "flow rate" (simulates continuous-time dynamics)
        self.rate = torch.nn.Parameter(torch.tensor(0.1))

    def model(self, x, y):
        # Sample weights from a time-evolving distribution
        weights = pyro.sample(
            "weights",
            dist.Normal(self.prior_mean, self.prior_var.sqrt()).to_event(2)
        )
        
        # Simulate flow process: Update distribution over `num_steps`
        for t in pyro.markov(range(self.num_steps)):
            # Likelihood (noise decreases over time)
            sigma = 1.0 / (1.0 + self.rate * t)  # Example noise schedule
            with pyro.plate("data", x.shape[0]):
                pred = x @ weights
                pyro.sample(f"obs_{t}", dist.Normal(pred, sigma), obs=y)

    def guide(self, x, y):
        # Variational posterior (track mean and variance over time)
        posterior_mean = pyro.param("posterior_mean", self.prior_mean.clone())
        posterior_var = pyro.param("posterior_var", self.prior_var.clone(), 
                                 constraint=dist.constraints.positive)
        pyro.sample("weights", dist.Normal(posterior_mean, posterior_var.sqrt()).to_event(2))

    def train(self, x_train, y_train, lr=0.01, epochs=1000):
        optimizer = Adam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        for epoch in range(epochs):
            loss = svi.step(x_train, y_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x_test, num_samples=100):
        # Sample from posterior to compute predictions with uncertainty
        posterior_mean = pyro.param("posterior_mean")
        posterior_var = pyro.param("posterior_var")
        
        with torch.no_grad():
            weights_samples = dist.Normal(posterior_mean, posterior_var.sqrt()).sample((num_samples,))
            preds = x_test @ weights_samples.permute(1, 0, 2)  # Shape: [num_samples, batch_size, output_dim]
        
        return preds.mean(0), preds.std(0)

# Example Usage
if __name__ == "__main__":
    # Toy data
    x_train = torch.randn(100, 1)
    y_train = 3 * x_train + torch.randn(100, 1) * 0.5
    
    # Train BFN
    bfn = BayesianFlowNetwork()
    bfn.train(x_train, y_train)
    
    # Predict
    x_test = torch.tensor([[0.5], [1.0]])
    mean, std = bfn.predict(x_test)
    print(f"Predictions (mean ± std):\n{mean}\n±\n{std}")