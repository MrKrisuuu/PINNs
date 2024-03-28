This is my master's thesis in the field of Computer Science at the Faculty of Computer Science at AGH. In it, I check how physics informed neural networks (PINNs) work in solving differential equations and how they maintain their invariants.

The best example is the Lotka-Volterra equation (predator-prey model), where adding an appropriate component to the loss function (Help loss) gives very good results.

```latex
\begin{align*}
    \frac{dx}{dt} &= (\alpha-\beta y)x \\
    \frac{dy}{dt} &= (\gamma x-\delta)y
\end{align*}

You can see this in the charts below.
