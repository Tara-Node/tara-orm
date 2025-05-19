#!/usr/bin/env bun

import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import os from "os";
import path from "path";
import ShortUniqueId from "short-unique-id";
import { z } from "zod";
import { mkdir } from "node:fs/promises";

import { dezerialize, type SzType, zerialize } from "zodex";
import dedent from "dedent";
import { OUTPUT_DATA_WRAPPER, RESERVED_SQLITE_WORDS } from "./constants";
import { measure, objToXml, validateSchemaMatch, validateSchemaOutput, verboseLog, zodToSqliteSchema } from "./helpers";

const { randomUUID: uuid } = new ShortUniqueId({
  length: 12,
  dictionary: "alphanum_lower",
});

interface AgentRecord {
  name: string;
  from_schema: string;
  to_schema: string;
  input_table: string;
  output_table: string;
}

type SQLiteValue = string | number | boolean | null | Uint8Array;

let db: Database;
{
  const dirname =
    import.meta.dir.startsWith("/$bunfs/root") ||
    import.meta.dir.startsWith("B:\\~BUN\\root")
      ? `${process.execPath}/..`
      : path.join(process.env.HOME!, ".mements");

  const dbPath = path.join(
    dirname,
    `${process.env.SATI_DB_NAME ?? "sati"}.sqlite`,
  );
  verboseLog("Database", "Initializing", { path: dbPath });

  await mkdir(path.dirname(dbPath), { recursive: true });
  db = new Database(dbPath);

  sqliteVec.load(db);

  if (os.platform() === "darwin") {
    Database.setCustomSQLite(
      "/opt/homebrew/Cellar/sqlite/3.47.0/lib/libsqlite3.0.dylib",
    );
  }

  db.query(`
    CREATE TABLE IF NOT EXISTS agents (
      name TEXT PRIMARY KEY,
      from_schema TEXT,
      to_schema TEXT,
      input_table TEXT,
      output_table TEXT
    )
  `).run();

  db.query<AgentRecord, any>(`
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(
      id TEXT PRIMARY KEY,
      agent_name TEXT NOT NULL PARTITION KEY,
      input_embedding FLOAT[1536],
      output_embedding FLOAT[1536]
    );
  `).run();
}

const systemMessage = dedent`
Instructions for Response Generation:

1. Analyze:
   - Use the provided {InputData} and {OutputSchema} to generate a {StructuredOutput}.

2. Format Response as XML:
   - The generated XML must be wrapped in a <${OUTPUT_DATA_WRAPPER}> tag.
   - The XML tags must exactly match the keys defined in the {OutputSchema}.

3. Rules:
   - The output must follow the output_schema exactly (including field names and types).
   - For arrays, use the structure: <array><item>value</item></array>.
   - For objects, each key should be represented as a nested tag with its value.
   - Escape special XML characters properly.
   - Do not include any extraneous XML tags beyond what is specified in the output schema.
`;

async function getEmbedding(text: string): Promise<number[]> {
  return measure(
    async () => {
      let headers: Record<string, string> = {
        "Content-Type": "application/json",
      };

      const embedEndpoint = process.env.EMBEDDINGS_API_URL
        ? `${process.env.EMBEDDINGS_API_URL}`
        : "https://api.mements.cloud/v1/embeddings";

      const response = await fetch(embedEndpoint, {
        method: "POST",
        headers: {
          ...headers,
          Authorization: `Bearer ${process.env.EMBEDDINGS_API_KEY}`,
        },
        body: JSON.stringify({
          model: "text-embedding-3-small",
          input: text,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        try {
          const error = JSON.parse(errorText);
          throw new Error(
            `Embedding API error: ${error.error?.message || errorText}`,
          );
        } catch {
          throw new Error(`Embedding API error: ${errorText}`);
        }
      }

      const data = await response.json();
      return data.data[0].embedding;
    },
    "Embedding",
    "Generate text embedding",
  );
}

async function callLLM(
  prompt: { system?: string; user: string },
  model: string,
  temperature: number = 0,
  reasoning_effort: string = "high",
) {
  return measure(
    async () => {
      const isClaude = model.includes("claude");
      const isGrok = model.includes("grok");
      const isDeepseek = model.includes("deepseek");
      const isGpt = model.includes("gpt");
      const isOx = model.startsWith("o1") || model.startsWith("o3");

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };

      const messages = [
        ...(prompt.system
          ? [{ role: isOx ? "developer" : "system", content: prompt.system }]
          : []),
        { role: "user" as const, content: prompt.user },
      ];

      let url = "";
      let body: Record<string, any> = {};

      if (isClaude) {
        headers["x-api-key"] = process.env.ANTHROPIC_API_KEY!;
        headers["anthropic-version"] = "2023-06-01";
        url = "https://api.anthropic.com/v1/messages";
        body = { model, max_tokens: 8000, messages };
      } else if (isGrok) {
        headers["Authorization"] = `Bearer ${process.env.GROK_API_KEY}`;
        url = "https://api.grok.x/v1/chat/completions";
        body = { model, messages };
      } else if (isDeepseek) {
        url = process.env.DEEPSEEK_API_URL
          ? process.env.DEEPSEEK_API_URL
          : "https://api.mements.cloud/v1/chat/completions";
        if (process.env.DEEPSEEK_API_KEY) {
          headers["Authorization"] = `Bearer ${process.env.DEEPSEEK_API_KEY}`;
        }
        body = { model, temperature, messages };
      } else if (isGpt || isOx) {
        headers["Authorization"] = `Bearer ${process.env.OPENAI_API_KEY}`;
        url = "https://api.openai.com/v1/chat/completions";
        if (isGpt) {
          body = { model, temperature, messages, max_tokens: 16000 };
        } else {
          body = { model, messages, max_completion_tokens: 16000, reasoning_effort };
        }
      } else {
        throw `Unexpected model: ` + model;
      }

      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorText = await response.text();

        try {
          const error = JSON.parse(errorText);
          throw new Error(
            `LLM API error: ${error.error?.message || errorText}`,
          );
        } catch {
          throw new Error(`LLM API error: ${errorText}`);
        }
      }

      const data = await response.json();
      // console.log("data ==> ", data);

      return isClaude ? data.content[0].text : data.choices[0].message.content;
    },
    "LLM",
    `Call ${model}`,
  );
}

type AgentInstance<I extends z.ZodObject<any>, O extends z.ZodObject<any>> = {
  exists: boolean;
  init: (from: I, to: O) => AgentInstance<I, O>;
  connectRemote: (url: string) => AgentInstance<I, O>;
  infer: (
    input: z.infer<I>,
    options?: { temperature?: number; model?: string; reasoning_effort?: string },
  ) => Promise<z.infer<O>>;
  recall: (
    input: z.infer<I> | null,
    output: z.infer<O> | null,
  ) => Promise<Array<{ id: string; input: z.infer<I>; output: z.infer<O>; distance: number }>>;
  reinforce: (input: z.infer<I>, output: z.infer<O>) => Promise<z.infer<O>>;
  find: (
    input?: Partial<z.infer<I>>,
    output?: Partial<z.infer<O>>,
  ) => Promise<Array<{ id: string; input: z.infer<I>; output: z.infer<O> }>>;
  store: (
    input: z.infer<I>,
    output: z.infer<O>,
  ) => Promise<{ id: string }>;
  addIndex: (
    table: "input" | "output",
    field: string,
  ) => Promise<void>;
  edit: (
    id: string,
    input: Partial<z.infer<I>>, output?: Partial<z.infer<O>>,
  ) => Promise<{ id: string; input: z.infer<I>; output: z.infer<O> }>;
  delete: (id: string) => Promise<{ id: string }>;
  erase: () => void;
};

function Agent<I extends z.ZodObject<any>, O extends z.ZodObject<any>>(name: string) {
  const sqliteNameRegex = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
  const sqliteReservedWords = new Set(RESERVED_SQLITE_WORDS);

  const isValid =
    sqliteNameRegex.test(name) &&
    !sqliteReservedWords.has(name.toUpperCase());

  if (!isValid) {
    throw new Error(
      `Invalid agent name: "${name}". Must be a valid SQLite table name.`,
    );
  }

  const logVerbose = (action: string, details: any) => {
    if (process.env.AGENT_MODE !== "silent") {
      if (typeof details == "object") {
        console.log(`[${name}] ${action}:`, JSON.stringify(details, null, 2));
      } else {
        console.log(`[${name}] ${action}:`, details);
      }
    }
  };

  let instance: ReturnType<typeof createAgentInstance> | null = null;
  let remoteUrl: string | null = null;

  const getOrCreateInstance = async () => {
    if (instance) return instance;

    const agent = db
      .query<AgentRecord, any>("SELECT * FROM agents WHERE name = ?")
      .get(name);

    if (!agent) {
      throw new Error("Agent not initialized");
    }

    instance = createAgentInstance(
      name,
      dezerialize(JSON.parse(agent.from_schema)) as I,
      dezerialize(JSON.parse(agent.to_schema)) as O,
      agent.input_table,
      agent.output_table,
      logVerbose,
    );

    return instance;
  };
  
  const makeRemoteRequest = async (endpoint: string, method: string, data: any) => {
    if (!remoteUrl) throw new Error("Remote connection not initialized");
    
    const response = await fetch(`${remoteUrl}${endpoint}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent: name, ...data }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Remote API error: ${errorText}`);
    }

    return response.json();
  };

  const checkRemoteExists = async (): Promise<boolean> => {
    if (!remoteUrl) return false;
    try {
      const response = await makeRemoteRequest('/exists', 'GET', {});
      return response.exists;
    } catch (error) {
      throw new Error(`Failed to check remote agent: ${error}`);
    }
  };

  const agentInterface: AgentInstance<I, O> = {
    get exists() {
      if (remoteUrl) {
        throw new Error("exists getter cannot be used with remote connection, use async check instead");
      }
      const agent = db
        .query<AgentRecord, any>("SELECT * FROM agents WHERE name = ?")
        .get(name);
      return !!agent;
    },

    connectRemote(url: string) {
      remoteUrl = url;
      return this;
    },

    init(from: I, to: O) {
      if (remoteUrl) {
        throw new Error("Cannot initialize local agent when connected remotely");
      }

      if (this.exists) {
        throw new Error("Agent already initialized");
      }

      const inputTable = `input_${name}`;
      const outputTable = `output_${name}`;

      db.query(zodToSqliteSchema(from, inputTable)).run();
      db.query(zodToSqliteSchema(to, outputTable)).run();

      logVerbose("Created tables", { inputTable, outputTable });

      db.query(`
        INSERT INTO agents (name, from_schema, to_schema, input_table, output_table)
        VALUES (?, ?, ?, ?, ?)
      `).run(
        name,
        JSON.stringify(zerialize(from)),
        JSON.stringify(zerialize(to)),
        inputTable,
        outputTable,
      );

      instance = createAgentInstance(
        name,
        from,
        to,
        inputTable,
        outputTable,
        logVerbose,
      );

      return this;
    },
    
    async infer(input: z.infer<I>, options) {
      try {
        if (remoteUrl) {
          return makeRemoteRequest('/infer', 'POST', { input, options });
        }

        const inst = await getOrCreateInstance();
        return inst.infer(input, options);
      } catch (error) {
        if (error instanceof z.ZodError) {
          const fieldErrors = error.issues
            .map(issue =>
              `Field '${issue.path.join(".")}': ${issue.message}`,
            )
            .join("\n");
          throw new Error(`Invalid input: \n${fieldErrors}`);
        }
        throw error;
      }
    },

    async recall(input: z.infer<I> | null, output: z.infer<O> | null) {
      if (remoteUrl) {
        return makeRemoteRequest('/recall', 'POST', { input, output });
      }      
      
      const inst = await getOrCreateInstance();
      return inst.recall(input, output);
    },
    async reinforce(input: z.infer<I>, output: z.infer<O>) {
      if (remoteUrl) {
        throw new Error("Cannot reinforce agent when connected remotely");
      }

      const inst = await getOrCreateInstance();
      return inst.reinforce(input, output);
    },

    async find(input?: Partial<z.infer<I>>, output?: Partial<z.infer<O>>) {
      if (remoteUrl) {
        return makeRemoteRequest('/find', 'POST', { input, output });
      }

      const inst = await getOrCreateInstance();
      return inst.find(input, output);
    },

    async store(input: z.infer<I>, output: z.infer<O>) {
      if (remoteUrl) {
        throw new Error("Cannot store data for agent when connected remotely");
      }

      const inst = await getOrCreateInstance();
      return inst.store(input, output);
    },

    async addIndex(table: "input" | "output", field: string) {
      if (remoteUrl) {
        throw new Error("Cannot add index for agent when connected remotely");
      }
      const inst = await getOrCreateInstance();
      return inst.addIndex(table, field);

    },

    async edit(id: string, input: Partial<z.infer<I>>, output?: Partial<z.infer<O>>) {
      const filters = { input, output };
      if (remoteUrl) {
        throw new Error("Cannot edit data for agent when connected remotely");
      }
      const inst = await getOrCreateInstance();
      return inst.edit(id, { input, output });
    },

    async delete(id: string) {
      if (remoteUrl) {
        throw new Error("Cannot delete data for agent when connected remotely");
      }
      const inst = await getOrCreateInstance();
      return inst.delete(id);
    },

    async erase() {
      if (remoteUrl) {
        throw new Error("Cannot delete agent when connected remotely");
      }
      
      const agent = db
        .query<AgentRecord, any>("SELECT * FROM agents WHERE name = ?")
        .get(name);

      if (!agent) {
        throw new Error("Agent not found");
      }

      const tx = db.transaction((name, input_table, output_table) => {
        db.query(`DROP TABLE IF EXISTS ${input_table}`).run();
        db.query(`DROP TABLE IF EXISTS ${output_table}`).run();
        db.query(`DELETE FROM vec_index WHERE agent_name = ?`).run(name);
        db.query(`DELETE FROM agents WHERE name = ?`).run(name);  
      })

      const it = tx(name, agent.input_table, agent.output_table);
      console.log("erased ==> ", name);

      // Reset instance
      instance = null;

      return { name };
    }
  };

  return agentInterface;
}

/**
 * Creates an agent instance with the specified schemas and configuration
 * @param name - Unique name for the agent
 * @param fromSchema - Zod schema defining input structure
 * @param toSchema - Zod schema defining output structure
 * @param inputTable - Name of table storing input data
 * @param outputTable - Name of table storing output data
 * @param logVerbose - Function for verbose logging
 * @returns Agent instance with inference and training capabilities
 */
function createAgentInstance(
  name: string,
  fromSchema: z.ZodObject<any>,
  toSchema: z.ZodObject<any>,
  inputTable: string,
  outputTable: string,
  logVerbose: (action: string, details: any) => void,
) {
  const existingAgent = db
    .query<AgentRecord, any>("SELECT * FROM agents WHERE name = ?")
    .get(name);

  if (existingAgent) {
    validateSchemaMatch(
      name,
      fromSchema,
      toSchema,
      existingAgent.from_schema,
      existingAgent.to_schema,
    );

  }

  const insertRecord = (table: string, data: Record<string, any>) => {
    return measure(
      async () => {
        const columns = Object.keys(data);
        const placeholders = columns.map(() => "?").join(", ");
        const values = Object.values(data).map(v => {
          if (typeof v === "object") return JSON.stringify(v);
          return v as SQLiteValue;
        });
        db.query(
          `INSERT INTO ${table} (${columns.join(", ")}) VALUES (${placeholders})`,
        ).run(...values);
      },
      "Database",
      `Insert into ${table}`,
    );
  };

  return {
    fromSchema,
    toSchema,
    inputTable,
    outputTable,
    async infer(
      input: z.infer<typeof fromSchema>,
      options: { temperature?: number; model?: string; reasoning_effort?: string } = {},
    ) {
      return measure(
        async () => {
          logVerbose("Infer: Received input", input);

          const validInput = fromSchema.parse(input);
          const inputXml = objToXml({ input_data: validInput });
          const schemaXml = objToXml({ output_schema: toSchema });

          const xmlString = `<root>${inputXml}${schemaXml}</root>`;
          logVerbose("Message to LLM", xmlString);

          const response = await callLLM(
            {
              system: systemMessage,
              user: xmlString,
            },
            options.model || "deepseek-ai/DeepSeek-R1",
            options.temperature,
            options.reasoning_effort || "high",
          );
          logVerbose("LLM Response", response);

          let modifiedResponse = response;
          if ("think" in toSchema.shape) {
            const it = (response: string): string => {
              const startTag = `<${OUTPUT_DATA_WRAPPER}>`;
              const endTag = `</${OUTPUT_DATA_WRAPPER}>`;

              const startIndex = response.indexOf(startTag);
              if (startIndex === -1) return response;

              let index = startIndex + startTag.length;
              let openCount = 1;
              let matchEndIndex = -1;

              while (index < response.length) {
                const nextStart = response.indexOf(startTag, index);
                const nextEnd = response.indexOf(endTag, index);

                if (nextEnd === -1) {
                  break;
                }

                if (nextStart !== -1 && nextStart < nextEnd) {
                  openCount++;
                  index = nextStart + startTag.length;
                } else {
                  openCount--;
                  index = nextEnd + endTag.length;
                  if (openCount === 0) {
                    matchEndIndex = nextEnd;
                    break;
                  }
                }
              }

              const outputDataContent = response.slice(
                startIndex + startTag.length,
                matchEndIndex,
              ).trim();

              if (outputDataContent.includes("<think>")) return response;

              const responseWithoutOutputData = response.slice(0, startIndex) +
                response.slice(matchEndIndex + endTag.length);

              const thinkRegex = /<think>([\s\S]*?)<\/think>/i;
              const externalThinkMatch = thinkRegex.exec(
                responseWithoutOutputData,
              );
              if (!externalThinkMatch) return response;

              const externalThinkContent = externalThinkMatch[1].trim();

              return `${startTag}${outputDataContent}<think>${externalThinkContent}</think>${endTag}`;
            };

            modifiedResponse = it(response);
          }

          const validation = validateSchemaOutput(modifiedResponse as string, toSchema);
          logVerbose("validation", validation);

          if (!validation.success) {
            throw `invalid llm output: ${validation.error}`;
          }

          return validation.data;
        },
        "Agent",
        `Inference for ${name}`,
      );
    },

    async recall(
      input: z.infer<typeof fromSchema> | null,
      output: z.infer<typeof toSchema> | null,
    ) {
      return measure(
        async () => {
          const target = input ? fromSchema.parse(input) : toSchema.parse(output);
          const targetXml = objToXml(target);
          const targetEmbed = await getEmbedding(targetXml); // todo: we should implement another table for caching these embeddings if its same targetXml we dont need to regenerate it just retrieve and it applies to recall as well to reinforce

          let query = dedent`
          WITH similars AS (
            SELECT id, distance
            FROM vec_index
            WHERE agent_name = ? AND ${input ? "input" : "output"}_embedding MATCH ?
            ORDER BY distance ASC
            LIMIT 10
          )
          SELECT i.*, o.*, si.distance, si.id
          FROM similars si
          LEFT JOIN ${inputTable} i ON i.id = si.id
          LEFT JOIN ${outputTable} o ON o.id = si.id
          WHERE 1=1
        `;

          const params: any[] = [name, JSON.stringify(targetEmbed)];
          logVerbose("Running similarity search", { query, name, embedSize: targetEmbed.length });

          const results = await measure(
            async () => db.query(query).all(...params),
            "Database",
            "Similarity search",
          );

          logVerbose("Resulting in", { results });
          return results.map((row: any) => {
            const processFields = (data: Record<string, any>) => {
              const processed: Record<string, any> = {};
              for (const [key, value] of Object.entries(data)) {
                if (key === "id" || key === "distance") {
                  processed[key] = value;
                  continue;
                }
                if (key.endsWith("_json")) {
                  const baseKey = key.slice(0, -5);
                  try {
                    processed[baseKey] = JSON.parse(value as string);
                  } catch {
                    processed[baseKey] = value;
                  }
                } else {
                  processed[key] = value;
                }
              }
              return processed;
            };

            try {
              const inputData: Record<string, any> = {};
              const outputData: Record<string, any> = {};
              const inputFields = new Set(Object.keys(fromSchema.shape));
              const outputFields = new Set(Object.keys(toSchema.shape));

              for (const [key, value] of Object.entries(row)) {
                const baseKey = key.endsWith("_json") ? key.slice(0, -5) : key;
                if (inputFields.has(baseKey)) {
                  inputData[key] = value;
                } else if (outputFields.has(baseKey)) {
                  outputData[key] = value;
                }
              }
              // logVerbose("inputData", inputData);
              // logVerbose("outputData", outputData);

              return {
                id: row.id,
                input: fromSchema.parse(processFields(inputData)),
                output: toSchema.parse(processFields(outputData)),
                distance: row.distance,
              };
            } catch (err) {
              logVerbose("Recall Error", err);
              return null;
            }
          }).filter((r: any) => r !== null);
        },
        "Agent",
        `Recall for ${name}`,
      );
    },

    async reinforce(input: any, output: any) {
      return measure(
        async () => {
          logVerbose("Training with input/output pair", { input, output });

          const validInput = fromSchema.parse(input);
          const validOutput = toSchema.parse(output);

          const qaId = uuid();

          const flatInput = Object.fromEntries(
            Object.entries(validInput).map(([k, v]) =>
              typeof v === "object"
                ? [`${k}_json`, JSON.stringify(v)]
                : [k, v]
            ),
          );
          const flatOutput = Object.fromEntries(
            Object.entries(validOutput).map(([k, v]) =>
              typeof v === "object"
                ? [`${k}_json`, JSON.stringify(v)]
                : [k, v]
            ),
          );

          await insertRecord(inputTable, { id: qaId, ...flatInput });
          await insertRecord(outputTable, { id: qaId, ...flatOutput });

          const inputXml = objToXml(validInput);
          const outputXml = objToXml(validOutput);

          const inputEmbed = await getEmbedding(inputXml);
          const outputEmbed = await getEmbedding(outputXml);

          db.query(`
            INSERT INTO vec_index (id, agent_name, input_embedding, output_embedding)
            VALUES (?, ?, ?, ?)
          `).run(
            qaId,
            name,
            JSON.stringify(inputEmbed),
            JSON.stringify(outputEmbed),
          );

          logVerbose("Stored training data", { qaId });

          return validOutput;
        },
        "Agent",
        `Reinforce for ${name}`,
      );
    },

    async store(
      input: z.infer<typeof fromSchema>,
      output: z.infer<typeof toSchema>,
    ) {
      return measure(
        async () => {
          logVerbose("Storing input/output pair", { input, output });

          const validInput = fromSchema.parse(input);
          const validOutput = toSchema.parse(output);

          const id = uuid();

          const flatInput = Object.fromEntries(
            Object.entries(validInput).map(([k, v]) =>
              typeof v === "object"
                ? [`${k}_json`, JSON.stringify(v)]
                : [k, v]
            ),
          );
          const flatOutput = Object.fromEntries(
            Object.entries(validOutput).map(([k, v]) =>
              typeof v === "object"
                ? [`${k}_json`, JSON.stringify(v)]
                : [k, v]
            ),
          );

          await insertRecord(inputTable, { id, ...flatInput });
          await insertRecord(outputTable, { id, ...flatOutput });

          logVerbose("Stored data with id", { id });
          return { id };
        },
        "Agent",
        `Store data for ${name}`,
      );
    },

    async find(
      input?: Partial<z.infer<typeof fromSchema>>,
      output?: Partial<z.infer<typeof toSchema>>,
    ) {
      return measure(
        async () => {
          logVerbose("Finding records with filters", { input, output });

          let query = dedent`
            SELECT i.*, o.*
            FROM ${inputTable} i
            JOIN ${outputTable} o ON o.id = i.id 
          `;

          const params: any[] = [];

          if (input) {
            for (const [key, value] of Object.entries(input)) {
              if (value !== undefined) {
                if (typeof value === "object") {
                  query += ` AND i.${key}_json = ?`;
                  params.push(JSON.stringify(value));
                } else {
                  query += ` AND i.${key} = ?`;
                  params.push(value);
                }
              }
            }
          }

          if (output) {
            for (const [key, value] of Object.entries(output)) {
              if (value !== undefined) {
                if (typeof value === "object") {
                  query += ` AND o.${key}_json = ?`;
                  params.push(JSON.stringify(value));
                } else {
                  query += ` AND o.${key} = ?`;
                  params.push(value);
                }
              }
            }
          }

          logVerbose("Running query", { query, params });

          const results = await measure(
            async () => db.query(query).all(...params),
            "Database",
            "Filter records",
          );

          logVerbose("Found records", { count: results.length });

          return results.map((row: any) => {
            const processFields = (data: Record<string, any>) => {
              const processed: Record<string, any> = {};
              for (const [key, value] of Object.entries(data)) {
                if (key === "id") {
                  processed[key] = value;
                  continue;
                }
                if (key.endsWith("_json")) {
                  const baseKey = key.slice(0, -5);
                  try {
                    processed[baseKey] = JSON.parse(value as string);
                  } catch {
                    processed[baseKey] = value;
                  }
                } else {
                  processed[key] = value;
                }
              }
              return processed;
            };

            try {
              const inputData: Record<string, any> = {};
              const outputData: Record<string, any> = {};
              const inputFields = new Set(Object.keys(fromSchema.shape));
              const outputFields = new Set(Object.keys(toSchema.shape));

              for (const [key, value] of Object.entries(row)) {
                const baseKey = key.endsWith("_json") ? key.slice(0, -5) : key;
                if (inputFields.has(baseKey)) {
                  inputData[key] = value;
                } else if (outputFields.has(baseKey)) {
                  outputData[key] = value;
                }
              }

              return {
                id: row.id,
                input: fromSchema.parse(processFields(inputData)),
                output: toSchema.parse(processFields(outputData)),
              };
            } catch (err) {
              logVerbose("Find Error", err);
              return null;
            }
          }).filter((r: any) => r !== null);
        },
        "Agent",
        `Find for ${name}`,
      );
    },

    async addIndex(table: "input" | "output", field: string) {
      return measure(
        async () => {
          const tableName = table === "input" ? inputTable : outputTable;
          const indexName = `idx_${tableName}_${field}`;
          
          logVerbose("Creating index", { table: tableName, field, indexName });

          try {
            db.query(`CREATE INDEX IF NOT EXISTS ${indexName} ON ${tableName}(${field})`).run();
            logVerbose("Index created successfully", { indexName });
          } catch (err) {
            logVerbose("Index creation failed", err);
            throw new Error(`Failed to create index: ${err}`);
          }
        },
        "Agent",
        `Create index for ${name}`,
      );  
    },

    async edit(
      id: string,
      patch: { input?: Partial<z.infer<typeof fromSchema>>; output?: Partial<z.infer<typeof toSchema>> },
    ) {
      return measure(
        async () => {
          logVerbose("Editing record", { id, patch });

          if (!patch.input && !patch.output) {
            throw new Error("No changes provided for edit.");
          }

          const row = db.query(`
                SELECT i.*, o.*
                FROM ${inputTable} i
                JOIN ${outputTable} o ON i.id = o.id
                WHERE i.id = ?
            `).get(id);

          if (!row) {
            throw new Error(`Record with id ${id} not found.`);
          }

          const inputFields = new Set(Object.keys(fromSchema.shape));
          const outputFields = new Set(Object.keys(toSchema.shape));
          const oldInput: Record<string, any> = {};
          const oldOutput: Record<string, any> = {};

          for (const [key, value] of Object.entries(row)) {
            const baseKey = key.endsWith("_json") ? key.slice(0, -5) : key;
            if (inputFields.has(baseKey)) {
              oldInput[baseKey] = value;
            } else if (outputFields.has(baseKey)) {
              oldOutput[baseKey] = value;
            }
          }

          const mergedInput = patch.input ? { ...oldInput, ...patch.input } : oldInput;
          const mergedOutput = patch.output ? { ...oldOutput, ...patch.output } : oldOutput;

          const validInput = fromSchema.parse(mergedInput);
          const validOutput = toSchema.parse(mergedOutput);

          const flatInput = Object.fromEntries(
            Object.entries(validInput).map(([k, v]) =>
              typeof v === "object" ? [`${k}_json`, JSON.stringify(v)] : [k, v]
            )
          );
          const flatOutput = Object.fromEntries(
            Object.entries(validOutput).map(([k, v]) =>
              typeof v === "object" ? [`${k}_json`, JSON.stringify(v)] : [k, v]
            )
          );

          const inputColumns = Object.keys(flatInput);
          const inputSetClause = inputColumns.map(col => `${col} = ?`).join(", ");
          const inputValues = Object.values(flatInput);
          db.query(`UPDATE ${inputTable} SET ${inputSetClause} WHERE id = ?`).run(...inputValues, id);

          const outputColumns = Object.keys(flatOutput);
          const outputSetClause = outputColumns.map(col => `${col} = ?`).join(", ");
          const outputValues = Object.values(flatOutput);
          db.query(`UPDATE ${outputTable} SET ${outputSetClause} WHERE id = ?`).run(...outputValues, id);

          const vecRow = db.query(
            `SELECT * FROM vec_index WHERE id = ? AND agent_name = ?`
          ).get(id, name);
          if (vecRow) {
            let newInputEmbed: number[];
            let newOutputEmbed: number[];

            if (patch.input) {
              const inputXml = objToXml(validInput);
              newInputEmbed = await getEmbedding(inputXml);
            } else {
              newInputEmbed = JSON.parse(vecRow.input_embedding);
            }

            if (patch.output) {
              const outputXml = objToXml(validOutput);
              newOutputEmbed = await getEmbedding(outputXml);
            } else {
              newOutputEmbed = JSON.parse(vecRow.output_embedding);
            }

            db.query(`
            UPDATE vec_index SET input_embedding = ?, output_embedding = ?
            WHERE id = ? AND agent_name = ?
          `).run(JSON.stringify(newInputEmbed), JSON.stringify(newOutputEmbed), id, name);
          }

          logVerbose("Record edited", { id, input: validInput, output: validOutput });
          return { id, input: validInput, output: validOutput };
        },
        "Agent",
        `Edit record for ${name}`,
      );
    },

    async delete(id: string) {
      return measure(
        async () => {
          logVerbose("Deleting record", { id });

          db.query(`DELETE FROM ${inputTable} WHERE id = ?`).run(id);
          db.query(`DELETE FROM ${outputTable} WHERE id = ?`).run(id);
          try {
            db.query(`DELETE FROM vec_index WHERE id = ? AND agent_name = ?`).run(id, name);
            logVerbose("Deleted record from vec_index", { id });
          } catch (e) {
            logVerbose("Vec index deletion error", { id, error: e });
          }
          return { id };
        },
        "Agent",
        `Delete record for ${name}`,
      );
    },    
  };
}

export { Agent };