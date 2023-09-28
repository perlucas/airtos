# Custom Trading Env Override for gym-anytrading

## Most relevant changes
- Agent input values are now normalized stock prices.
- Added a new "Noop" Action for the agent to choose.
- Agent is now expected to produce an additional `StrengthIndicator` output.
- Agent is now given the possibility to trade more than 1 stock in a single transaction.

## Environment inputs
An array of length `window_size`, each input is a number in the range (0, 1], representing normalized stock prices (1 = Max possible stock price for frame).

## Environment Outputs
Agent is now expected to produce 2 output values. First one will be `Action`, which can be either "Sell", "Buy" or "Noop" (new). Second will be `StrengthIndicator`, which can be either "Weak", "Regular" or "Strong".

The `StrengthIndicator` value will be used to determine number of stocks to be traded by the agent in a specific transaction. The stronger the indicator, the more stocks will be traded.



# TODOs
- Renderizar bien las posiciones, sólamente cuando el bot vaya a operar.
- Asignar una variación de colores según el Strength
- Revisar el algoritmo, armarlo en base a los precios. Calcular el presupuesto inicial según el 1er precio.
- Mejorar las variables mostradas al testear (gráfico).