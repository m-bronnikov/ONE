## Reduce constant duplication via castom source

Let's refactor GraphBuilderSource and delegate embedded constants building to castom GraphBuilderSource inheritor.

Plan:

1. Refactor GraphBuilderSource in [luci]
2. Add required [luci-micro-extension] as separate module
3. Add support of new node to [luci-interpeter]
4. Provide example
