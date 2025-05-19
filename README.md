A type-safe persistence layer for AI agents with semantic memory and structured I/O.

```typescript
import { Agent } from '@mements/tara-orm';
import { z } from 'zod';

const ticketSchema = z.object({ issue: z.string() });
const responseSchema = z.object({ solution: z.string() });
const supportAgent = Agent('support').init(ticketSchema, responseSchema);
```

## Quick Start

```bash
# Install the package
npm install @mements/tara-orm zod

# Set up your environment variables
echo "EMBEDDINGS_API_KEY=your_key" > .env
echo "ANTHROPIC_API_KEY=your_key" >> .env  # Or any other LLM provider
```

```typescript
import { Agent } from '@mements/tara-orm';
import { z } from 'zod';

// 1. Define your schemas
const inputSchema = z.object({
  question: z.string(),
  context: z.string().optional()
});

const outputSchema = z.object({
  answer: z.string(),
  sources: z.array(z.string()).optional()
});

// 2. Initialize your agent
const agent = Agent('my_agent').init(inputSchema, outputSchema);

// 3. Use your agent
const result = await agent.infer({
  question: "What is tara-orm?",
  context: "I'm building an AI application"
});

console.log(result);
// { answer: "tara-ORM is a type-safe persistence layer...", sources: [...] }
```

## System Architecture

tara-ORM creates a powerful dual-table architecture with vector embeddings:

```
┌───────────────────┐       ┌────────────────────┐
│    Input Table    │       │    Output Table    │
├───────────────────┤       ├────────────────────┤
│ id: string (PK)   │───┐   │ id: string (PK)    │
│ field1: type      │   └──>│ field1: type       │
│ field2: type      │       │ field2: type       │
│ object_json: text │       │ object_json: text  │
└───────────────────┘       └────────────────────┘
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │     Vector Index     │
              ├──────────────────────┤
              │ id: string           │
              │ agent_name: string   │
              │ input_embedding: vec │
              │ output_embedding: vec│
              └──────────────────────┘
```

### How It Works

When you create and use an agent, here's what happens under the hood:

1. **Schema Translation**: Your Zod schemas are converted to SQLite tables
2. **Data Storage**: Input/output pairs are stored in their respective tables
3. **Vector Embedding**: Text is converted to vector embeddings for semantic search
4. **LLM Formatting**: Data is formatted as XML for consistent LLM responses
5. **Type Validation**: All data is validated against your schemas

## RAG-Enabled Customer Support Bot

Let's build a complete customer support agent with retrieval-augmented generation:

```typescript
import { Agent } from '@mements/tara-orm';
import { z } from 'zod';

// Define ticket schema with customer data and convertaraon history
const ticketSchema = z.object({
  customer: z.object({
    name: z.string(),
    email: z.string().email(),
    tier: z.enum(['free', 'pro', 'enterprise'])
  }),
  issue: z.string().describe('Customer problem description'),
  category: z.enum(['billing', 'technical', 'account']).optional(),
  // This field will store similar past tickets for RAG
  similarIssues: z.array(z.object({
    issue: z.string(),
    solution: z.string()
  })).optional()
});

// Define response schema
const responseSchema = z.object({
  solution: z.string().describe('Response to customer'),
  internalNotes: z.string().describe('Notes for support team'),
  nextSteps: z.array(z.string()),
  category: z.enum(['billing', 'technical', 'account'])
});

// Create and initialize the agent
const supportBot = Agent('support_bot').init(ticketSchema, responseSchema);
```

### What Happens When We Initialize the Agent?

When `init()` is called, the following database tables are created:

```
┌───────────────────────────┐      ┌─────────────────────────────┐
│ input_support_bot (table) │      │ output_support_bot (table)  │
├───────────────────────────┤      ├─────────────────────────────┤
│ id: TEXT PRIMARY KEY      │      │ id: TEXT PRIMARY KEY        │
│ customer_json: TEXT       │      │ solution: TEXT              │
│ issue: TEXT               │      │ internalNotes: TEXT         │
│ category: TEXT            │      │ nextSteps_json: TEXT        │
│ similarIssues_json: TEXT  │      │ category: TEXT              │
└───────────────────────────┘      └─────────────────────────────┘
```

Notice how:
- Complex objects like `customer` become JSON fields (`customer_json`)
- Arrays like `nextSteps` become JSON fields (`nextSteps_json`)
- Simple fields remain as their respective SQL types

### Implementing RAG Workflow

Now let's implement a complete RAG workflow:

```typescript
// Process a new customer ticket with RAG
async function handleTicket(ticketData) {
  // 1. Extract the issue
  const { issue } = ticketData;
  
  // 2. Search for similar past tickets using vector similarity
  const similarTickets = await supportBot.recall({ issue }, null);
  console.log(`Found ${similarTickets.length} similar tickets`);
  
  // Diagram of what happens during recall():
  // 
  // ┌──────────┐    ┌─────────────┐    ┌──────────────┐
  // │  Input   │───>│  Generate   │───>│Input Embedding│
  // │ (issue)  │    │  Embedding  │    │   (vector)    │
  // └──────────┘    └─────────────┘    └──────────────┘
  //                                            │
  //                                            ▼
  // ┌──────────────────┐    ┌─────────────────────────────┐
  // │ Similar Records  │<───│ Vector Similarity Search    │
  // │ (sorted by      │    │ in vec_index table           │
  // │  similarity)    │    │ using input_embedding MATCH  │
  // └──────────────────┘    └─────────────────────────────┘
  
  // 3. Extract relevant context from similar tickets
  const relevantTickets = similarTickets.slice(0, 3).map(ticket => ({
    issue: ticket.input.issue,
    solution: ticket.output.solution
  }));
  
  // 4. Create augmented ticket with RAG context
  const augmentedTicket = {
    ...ticketData,
    similarIssues: relevantTickets
  };
  
  // 5. Generate response using augmented context
  const response = await supportBot.infer(augmentedTicket, {
    temperature: 0.3,  // Lower for more consistent support responses
    model: "claude-3-opus-20240229"
  });
  
  // Diagram of what happens during infer():
  //
  // ┌──────────────┐    ┌─────────────┐    ┌───────────────┐
  // │ Augmented    │───>│ Convert to  │───>│ Send to LLM   │
  // │ Ticket Data  │    │ XML Format  │    │ with Schemas  │
  // └──────────────┘    └─────────────┘    └───────────────┘
  //                                                │
  //                                                ▼
  // ┌──────────────┐    ┌─────────────┐    ┌───────────────┐
  // │ Validated    │<───│ Parse XML   │<───│ LLM Response  │
  // │ Response     │    │ Response    │    │ as XML        │
  // └──────────────┘    └─────────────┘    └───────────────┘
  
  // 6. Store this interaction for future reference
  await supportBot.reinforce(augmentedTicket, response);
  
  // Diagram of what happens during reinforce():
  //
  // ┌──────────┐    ┌──────────────┐    ┌────────────────┐
  // │ Input &  │───>│ Generate     │───>│ Input & Output │
  // │ Output   │    │ Embeddings   │    │ Embeddings     │
  // └──────────┘    └──────────────┘    └────────────────┘
  //       │                                     │
  //       ▼                                     ▼
  // ┌─────────────────┐              ┌─────────────────────┐
  // │ Store in Input  │              │ Store in vec_index  │
  // │ & Output Tables │              │ for future recall   │
  // └─────────────────┘              └─────────────────────┘
  
  return response;
}

// Example usage
const ticket = {
  customer: {
    name: "Jordan Smith",
    email: "jordan@example.com",
    tier: "pro"
  },
  issue: "I can't export my data to CSV. The export button is disabled.",
  category: "technical"
};

const response = await handleTicket(ticket);
console.log(response);
```

### Updating Ticket Data

When the customer provides additional information, you can update the record:

```typescript
// Update the ticket with new information
async function updateTicket(id, newData) {
  // Edit stored record
  await supportBot.edit(id, newData);
  
  // Diagram of what happens during edit():
  //
  // ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  // │ Record ID &  │───>│ Retrieve     │───>│ Update       │
  // │ New Data     │    │ Existing Data│    │ Database     │
  // └──────────────┘    └──────────────┘    └──────────────┘
  //                                                │
  //                                                ▼
  // ┌──────────────────┐             ┌────────────────────────┐
  // │ Update Vector    │<────────────│ Generate New Embeddings│
  // │ Index            │             │ For Changed Fields     │
  // └──────────────────┘             └────────────────────────┘
}
```

## API Reference

### Agent Creation and Management

```typescript
// Create or access an agent
Agent(name: string)

// Initialize with schemas  
agent.init(inputSchema: ZodObject, outputSchema: ZodObject)

// Connect to remote agent
agent.connectRemote(url: string)

// Delete agent and all data
agent.erase()
```

### Core Operations

```typescript
// Generate output using LLM
await agent.infer(
  input: YourInputType,
  options?: {
    temperature?: number;   // 0-1, controls randomness
    model?: string;         // e.g., "claude-3-opus-20240229"
    reasoning_effort?: string; // For models that support it
  }
)

// Find similar past interactions
await agent.recall(
  input: YourInputType | null,
  output: YourOutputType | null
)

// Store with vector embeddings
await agent.reinforce(input, output)

// Store without embeddings
await agent.store(input, output)
```

### Data Management

```typescript
// Query stored data by exact match
await agent.find(inputFilter?, outputFilter?)

// Create index for faster queries
await agent.addIndex("input" | "output", fieldName)

// Update stored record
await agent.edit(id, inputUpdates?, outputUpdates?)

// Delete record
await agent.delete(id)
```

## Data Flow Diagram

```
┌─────────────┐     ┌────────────┐     ┌───────────────┐
│ Define Zod  │────>│ Initialize │────>│ SQLite Tables │
│ Schemas     │     │ Agent      │     │ Created       │
└─────────────┘     └────────────┘     └───────────────┘
                                               │
      ┌────────────────────────────────────────┘
      │
      ▼
┌─────────────┐     ┌────────────┐     ┌───────────────┐
│ Agent.infer │────>│ LLM Call   │────>│ XML Response  │
│ (input)     │     │ with XML   │     │ Parsed        │
└─────────────┘     └────────────┘     └───────────────┘
      │                                        │
      │                                        ▼
      │                               ┌───────────────┐
      │                               │ Validated     │
      │                               │ Output        │
      │                               └───────────────┘
      │                                        │
      ▼                                        ▼
┌─────────────┐     ┌────────────┐     ┌───────────────┐
│ Agent.      │────>│ Store Data │────>│ Generate      │
│ reinforce   │     │ in Tables  │     │ Embeddings    │
└─────────────┘     └────────────┘     └───────────────┘
                                               │
                                               ▼
                                      ┌───────────────┐
                                      │ Store in      │
                                      │ Vector Index  │
                                      └───────────────┘
```

## Environment Configuration

```typescript
// Required for embeddings
EMBEDDINGS_API_KEY=your_key
EMBEDDINGS_API_URL=https://api.example.com/embeddings  // Optional

// At least one LLM provider required
ANTHROPIC_API_KEY=your_key    // For Claude
OPENAI_API_KEY=your_key       // For GPT models
DEEPSEEK_API_KEY=your_key     // For DeepSeek models
GROK_API_KEY=your_key         // For Grok models

// Optional configuration
AGENT_MODE=silent             // Suppress logs
tara_DB_NAME=custom_db_name   // Custom database name
VERBOSE_MODE=true             // Enable detailed logging
```

## Supported LLM Models

| Provider | Model Examples                  | Environment Variable  |
|----------|--------------------------------|----------------------|
| Claude   | claude-3-opus-20240229        | ANTHROPIC_API_KEY    |
| GPT      | gpt-4-turbo                   | OPENAI_API_KEY       |
| DeepSeek | deepseek-ai/DeepSeek-R1       | DEEPSEEK_API_KEY     |
| Grok     | grok-1                        | GROK_API_KEY         |

---
