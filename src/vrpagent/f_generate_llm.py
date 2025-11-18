"""
LLM Manager for VRPAGENT
Handles communication with LLM APIs (OpenAI, Claude, etc.)
"""

class LLMManager:
    """
    Manager for LLM interactions
    Handles prompt generation and code generation
    """
    
    def __init__(self, provider='openai', model='gpt-4', api_key=None):
        """
        Initialize LLM Manager
        
        Args:
            provider: 'openai' or 'claude'
            model: Model name (e.g., 'gpt-4', 'claude-3-opus')
            api_key: API key (if None, reads from environment)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize API client based on provider"""
        if self.provider == 'openai':
            import openai
            return openai.OpenAI(api_key=self.api_key)
        elif self.provider == 'claude':
            from anthropic import Anthropic
            return Anthropic(api_key=self.api_key)
        elif self.provider == 'groq':
            import openai
            # Groq uses OpenAI-compatible API
            return openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def Generate(self, prompt, temperature=0.8, max_tokens=2000):
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        if self.provider in ['openai', 'groq']:
            # Both OpenAI and Groq use same API format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in operations research and VRP heuristic design."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == 'claude':
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are an expert in operations research and VRP heuristic design.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
    
    def GenerateBatch(self, prompts, **kwargs):
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of prompts
            **kwargs: Arguments for Generate()
        
        Returns:
            List of generated texts
        """
        return [self.Generate(prompt, **kwargs) for prompt in prompts]
    
    def f_generate_llm(self, operator_type='f_remove', examples=None, temperature=0.8):
        """
        Generate VRP heuristic operator using LLM
        
        Args:
            operator_type: 'f_remove' (destroy) or 'f_order' (ordering)
            examples: Optional list of example heuristics for inspiration
            temperature: Sampling temperature for generation
        
        Returns:
            Tuple of (heuristic_name, heuristic_code, compiled_function)
        """
        if operator_type == 'f_remove':
            prompt = self._create_destroy_prompt(examples)
        elif operator_type == 'f_order':
            prompt = self._create_ordering_prompt(examples)
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")
        
        # Generate code from LLM
        response = self.Generate(prompt, temperature=temperature)
        
        # Parse response
        name, code = self._parse_llm_response(response)
        
        # Validate and compile
        compiled_func = self._validate_and_compile(code, operator_type)
        
        return name, code, compiled_func
    
    def _create_destroy_prompt(self, examples=None):
        """Create prompt for generating f_remove (destroy operator)"""
        prompt = """You are an expert in operations research and VRP heuristic design.

Design a DESTROY OPERATOR (f_remove) for the Vehicle Routing Problem using Large Neighborhood Search.

A destroy operator selects customers to remove from the current solution. Good destroy operators should:
1. Remove customers that can lead to better solutions when reinserted
2. Balance diversification (exploring new areas) and intensification (improving current areas)
3. Consider spatial proximity, route structure, and customer characteristics

## CRITICAL DATA STRUCTURE INFORMATION:
- **solution.routes** is a List[List[int]] - each route contains customer IDs (integers 1, 2, 3, ...)
- **instance.customers** is a List[Customer] - indexed from 0
- To get Customer object from ID: `customer_obj = instance.customers[customer_id - 1]`
- Customer IDs start at 1, but list indices start at 0!

## Function Signature:
```python
def f_remove(instance, solution, num_remove):
    '''
    Remove customers from solution
    
    Args:
        instance: VRP problem instance
        solution: Current solution 
        num_remove: Number of customers to remove
    
    Returns:
        (partial_solution, removed_customers) - both solutions have routes as List[List[int]]
    '''
    import random
    import copy
    
    # IMPORTANT: Work with customer IDs (integers), not Customer objects
    partial_solution = copy.deepcopy(solution)
    removed_customers = []
    
    # Get all customer IDs from all routes
    all_customer_ids = [cid for route in partial_solution.routes for cid in route]
    
    # Your removal logic here
    # Example: Remove random customers
    customers_to_remove = random.sample(all_customer_ids, min(num_remove, len(all_customer_ids)))
    
    # Remove from routes
    for customer_id in customers_to_remove:
        for route in partial_solution.routes:
            if customer_id in route:
                route.remove(customer_id)
                removed_customers.append(customer_id)
                break
    
    partial_solution.invalidate_objective()
    return (partial_solution, removed_customers)
```

## Available Attributes and Methods:
```python
# Instance attributes:
instance.customers          # List[Customer] (0-indexed)
instance.depot             # Depot object with .x, .y
instance.get_distance(from_id, to_id)  # Distance between nodes (depot=0, customers=1,2,3...)
instance.vehicle_capacity  # Max capacity per vehicle
instance.num_customers     # Total number of customers

# Customer attributes (get via: instance.customers[id-1]):
customer.id       # int (1, 2, 3, ...)
customer.x, customer.y    # float coordinates
customer.demand   # float demand value

# Solution attributes:
solution.routes   # List[List[int]] - e.g., [[1,3,5], [2,4], [6,7,8]]
solution.objective  # float - total route distance
solution.copy()   # Create deep copy
solution.invalidate_objective()  # Call after modifying routes
```

## COMMON MISTAKES TO AVOID:
❌ `for customer in solution.routes[0]:` then `customer.x` - customer is an INT!
❌ `routes[customer]` - routes needs integer index, not customer object
❌ `instance.customers[customer_id]` - OFF BY ONE! Should be `[customer_id - 1]`
✅ `customer_obj = instance.customers[cid - 1]` then `customer_obj.x`
✅ `for customer_id in solution.routes[0]:` - customer_id is an int

## Requirements:
1. Return EXACTLY num_remove customers (or fewer if not enough available)
2. Return tuple: (partial_solution, list_of_removed_customer_IDs)
3. Both solutions must have routes as List[List[int]]
4. Call partial_solution.invalidate_objective() before returning
5. Import any needed modules (random, copy, math, etc.) inside the function

Provide:
**Heuristic Name:** [Short descriptive name]

**Python Code:**
```python
def f_remove(instance, solution, num_remove):
    # Your implementation
    pass
```
"""

        if examples:
            prompt += "\n## Examples for Inspiration (DO NOT COPY):\n"
            for i, ex in enumerate(examples, 1):
                prompt += f"\n### Example {i}:\n```python\n{ex}\n```\n"
        
        return prompt
    
    def _create_ordering_prompt(self, examples=None):
        """Create prompt for generating f_order (ordering operator)"""
        prompt = """You are an expert in operations research and VRP heuristic design.

Design an ORDERING OPERATOR (f_order) for the Vehicle Routing Problem using Large Neighborhood Search.

An ordering operator determines the sequence in which removed customers should be reinserted. Good ordering strategies can:
1. Prioritize customers that are hard to insert
2. Consider regret (difference between best and second-best insertion)
3. Balance greediness with flexibility

## Function Signature:
```python
def f_order(instance, partial_solution, removed_customers):
    '''
    Determine reinsertion order for removed customers
    
    Args:
        instance: VRP problem instance
        partial_solution: Solution with customers removed
        removed_customers: List of removed customer IDs
    
    Returns:
        List of customer IDs in insertion order
    '''
    # Your implementation here
    return ordered_customers
```

## Available Attributes:
- instance.customers: List of customers
- instance.depot: Depot location
- instance.get_distance(i, j): Distance between nodes
- partial_solution.routes: Current routes
- customer.x, customer.y: Coordinates
- customer.demand: Demand value

## Common Strategies:
- Random order
- By distance to depot
- By regret value (difference between best and 2nd-best insertion cost)
- By demand (largest/smallest first)
- By difficulty to insert

## Requirements:
1. Return ALL customers in removed_customers (in some order)
2. Must be a valid Python function
3. Be creative but practical

Provide:
**Heuristic Name:** [Name]

**Python Code:**
```python
# Your implementation
```
"""

        if examples:
            prompt += "\n## Examples for Inspiration (DO NOT COPY):\n"
            for i, ex in enumerate(examples, 1):
                prompt += f"\n### Example {i}:\n```python\n{ex}\n```\n"
        
        return prompt
    
    def _parse_llm_response(self, response):
        """Parse LLM response to extract name and code"""
        import re
        
        # Extract name
        name_match = re.search(r'\*\*Heuristic Name:\*\*\s*(.+?)(?:\n|$)', response)
        if name_match:
            name = name_match.group(1).strip()
        else:
            name = "GeneratedHeuristic"
        
        # Extract code
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if not code_match:
            raise ValueError("No Python code block found in LLM response")
        
        code = code_match.group(1).strip()
        
        return name, code
    
    def _validate_and_compile(self, code, operator_type):
        """Validate and compile generated code"""
        import ast
        
        try:
            # Parse to check syntax
            ast.parse(code)
            
            # Create namespace and execute
            namespace = {}
            exec(code, namespace)
            
            # Find the function
            if operator_type == 'f_remove':
                func_names = ['f_remove', 'destroy', 'remove_customers']
            else:  # f_order
                func_names = ['f_order', 'order', 'ordering']
            
            compiled_func = None
            for func_name in func_names:
                if func_name in namespace and callable(namespace[func_name]):
                    compiled_func = namespace[func_name]
                    break
            
            if compiled_func is None:
                raise ValueError(f"No valid {operator_type} function found in generated code")
            
            return compiled_func
            
        except Exception as e:
            print(f"Validation failed: {str(e)}")
            return None

