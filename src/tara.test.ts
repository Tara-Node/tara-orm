import { afterAll, beforeAll, describe, expect, it } from "bun:test";

import { Agent } from './tara';
import { z } from "zod";
import dedent from "dedent";
import type { Server } from "bun";
import { zerialize } from "zodex";

let server: Server;

const mockValues = {
    think: `I understood "btc" to refer to Bitcoin, so I gathered concise facts on its decentralized, blockchain-based nature along with its historical significance.`,
    conclusion: `Bitcoin (BTC) is a decentralized digital currency created in 2009 by the pseudonymous Satoshi Nakamoto, operating on blockchain technology and noted for its capped supply and role as digital gold.`,
    query: "bitcoin price",
    explanation: "need to know it",
}

beforeAll(() => {
    server = Bun.serve({
        port: 0,
        fetch(req) {
            const content = dedent`
<output_data>
    <think>${mockValues.think}</think>
    <response>
        <conclusion>${mockValues.conclusion}</conclusion>
        <additionalQueries>
            <item>
                <query>${mockValues.query}</query>
                <explanation>${mockValues.explanation}</explanation>
            </item>
        </additionalQueries>
    </response>
</output_data>
            `;
            return new Response(
                JSON.stringify({ choices: [{ message: { content } }] })
            )
        }
    });
    process.env.DEEPSEEK_API_URL = `http://localhost:${server.port}`;
    process.env.VERBOSE_MODE = 'true';
});

afterAll(() => {
    server?.stop(true);
});

it('safe-agents', async () => {
    const schemaIn = z.object({
        question: z.string().describe('qwerqwer'),
        context: z.array(z.string()).optional().describe("more about"),
        something: z.string().optional().describe("else"),
    });
    
    const schemaOut = z.object({
        response: z.object({
            conclusion: z.string().describe('short, fit in a tweet'),
            additionalQueries: z.array(z.object({
                query: z.string(),
                explanation: z.string(),
            })).optional().describe("facts"),
        }),
        think: z.string()
    });
    
    const myFirstAgent = Agent<typeof schemaIn, typeof schemaOut>("first");

    if (myFirstAgent.exists) {
    console.log("myFirstAgent.exists ==> ", myFirstAgent.exists);
        myFirstAgent.erase();
        console.log("myFirstAgent.erase(); ==> ");
    }

    myFirstAgent.init(schemaIn, schemaOut);
    
    const result = await myFirstAgent.infer({
        question: 'asdfasfd',
        context: ['first', 'second'],
    }, {
        model: 'deepseek-ai/DeepSeek-R1',
    });

    expect(result.think).toBe(mockValues.think);
    expect(result.response.conclusion).toBe(mockValues.conclusion);
    expect(result.response.additionalQueries[0].query).toBe(mockValues.query);
    expect(result.response.additionalQueries[0].explanation).toBe(mockValues.explanation);
});