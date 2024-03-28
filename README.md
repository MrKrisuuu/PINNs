This is my master's thesis in the field of Computer Science at the Faculty of Computer Science at AGH. In it, I check how physics informed neural networks (PINNs) work in solving differential equations and how they maintain their invariants.

The best example is the Lotka-Volterra equation (predator-prey model), where adding an appropriate component to the loss function (Help loss) gives very good results.

```math
\begin{align*}
    \frac{dx}{dt} &= (\alpha-\beta y)x \\
    \frac{dy}{dt} &= (\gamma x-\delta)y
\end{align*}
```
These equations hold:
```math
\alpha ln(y) + \beta y + \gamma x - \delta ln(x) = Const
````

Training and comparison with numerical methods (step is h=0.001) are below:
<p align="center">
  <img src="[https://picsum.photos/460/300](https://github.com/MrKrisuuu/PINNs/assets/92759002/f410b055-a01c-4e25-b0e0-031eb77cd917)">
</p>
![LV_help_loss](https://github.com/MrKrisuuu/PINNs/assets/92759002/f410b055-a01c-4e25-b0e0-031eb77cd917)
![LV_help_pinn_result](https://github.com/MrKrisuuu/PINNs/assets/92759002/9b6b7a9e-69a4-493e-bb7b-e4cd13399dfa)
![LV_help_predator](https://github.com/MrKrisuuu/PINNs/assets/92759002/17948b4f-e4fc-4b96-a8ad-26afdcd51884)
![LV_help_prey](https://github.com/MrKrisuuu/PINNs/assets/92759002/ec115873-a7bf-45c9-b1db-22e20c2414c9)
![LV_help_constant](https://github.com/MrKrisuuu/PINNs/assets/92759002/84769b19-1b9f-4146-a174-6a33c0b6f954)



