from binance.client import Client
from binance.enums import SIDE_SELL, SIDE_BUY

# Replace with your API keys
# ifa02
#api_key = ""
#secret_key = ""
# bnc04
api_key = ""
secret_key = ""

# Initialize the Binance client
client = Client(api_key, secret_key)

# Function to fetch the maximum allowable order quantity for a symbol
def get_max_quantity(symbol):
    try:
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                for filter in s['filters']:
                    if filter['filterType'] == 'LOT_SIZE':
                        max_qty = float(filter['maxQty'])
                        print(f"Max quantity for {symbol}: {max_qty}")  # Debug statement
                        return max_qty
        print(f"Symbol {symbol} not found in exchange info.")
    except Exception as e:
        print(f"Error fetching max quantity for {symbol}: {e}")
    return None

# Function to close all futures positions
def close_all_futures_positions():
    try:
        # Fetch all open futures positions
        positions = client.futures_position_information()

        for position in positions:
            symbol = position['symbol']
            position_amt = abs(float(position['positionAmt']))

            if position_amt != 0:  # If there is an open position
                # Determine the side to close the position (sell if long, buy if short)
                side = SIDE_SELL if float(position['positionAmt']) > 0 else SIDE_BUY

                print(f"Closing position for {symbol}: {position_amt}")

                # Get the max allowable order quantity for the symbol
                max_qty = get_max_quantity(symbol)
                if not max_qty:
                    print(f"Unable to fetch max quantity for {symbol}. Skipping...")
                    continue

                # Split the position into smaller orders if needed
                while position_amt > 0:
                    order_qty = min(position_amt, max_qty)  # Use the smaller of remaining position or max_qty
                    print(f"Placing order for {symbol}: {order_qty}")  # Debug statement

                    try:
                        # Create a market order with 'reduce only' option
                        client.futures_create_order(
                            symbol=symbol,
                            side=side,
                            type='MARKET',
                            quantity=order_qty,
                            reduceOnly=True
                        )
                    except Exception as e:
                        print(f"Error placing order for {symbol}: {e}")
                        break  # Stop further attempts if an error occurs

                    position_amt -= order_qty  # Reduce the remaining position amount
                    print(f"Remaining position for {symbol}: {position_amt}")  # Debug statement
        
        print("All positions closed.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to close all positions
close_all_futures_positions()
