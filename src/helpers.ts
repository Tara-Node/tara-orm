import chalk from "chalk";
import dedent from "dedent";
import { z } from "zod";
import { OUTPUT_DATA_WRAPPER } from "./constants";
import { dezerialize, zerialize, zerialize } from "zodex";

export const verboseLog = (category: string, action: string, details?: any) => {
  const isVerbose = process.env.VERBOSE_MODE !== undefined
    ? process.env.VERBOSE_MODE === "true"
    : import.meta.main;

  if (isVerbose) {
    const categoryStr = chalk.blue(`[${category}]`);
    const actionStr = chalk.green(action);

    let detailsStr = "";
    if (details) {
      if (typeof details === "string" && details.includes("ms")) {
        detailsStr = chalk.gray(` (${details})`);
      } else if (details instanceof Error) {
        console.error(details);
        detailsStr = `${details}`;
      } else {
        const indent = " ".repeat(category.length + 3);
        const formattedDetails = typeof details === "string"
          ? details
          : JSON.stringify(details, null, 2)
              .split("\n")
              .join(`\n${indent}`);
        detailsStr = `\n${indent}${chalk.gray(formattedDetails)}`;
      }
    }

    const message = `${categoryStr} ${actionStr}${detailsStr}`;

    if (action?.startsWith("Starting") || action?.startsWith("Completed")) {
      console.log("");
    }

    console.log(message);
  }
};

export async function measure<T>(
  fn: () => Promise<T>,
  category: string,
  action: string,
): Promise<T> {
  const start = performance.now();
  try {
    verboseLog(category, `Starting ${action}`);
    const result = await fn();
    const duration = performance.now() - start;
    verboseLog(category, `Completed ${action}`, `${duration.toFixed(2)}ms`);
    return result;
  } catch (error) {
    verboseLog(category, `Failed ${action}`, error);
    throw `${category} ${action} failed`;
  }
}

/**
 * Converts a Zod schema to SQLite table schema
 * @param schema - Zod object schema
 * @param tableName - Name for the SQLite table
 * @returns SQLite CREATE TABLE statement
 */
export function zodToSqliteSchema(
  schema: z.ZodObject<any>,
  tableName: string,
): string {
  const columns = [];
  for (const [key, value] of Object.entries(schema.shape)) {
    if (value._def.typeName === "ZodString") {
      columns.push(`${key} TEXT`);
    } else if (value._def.typeName === "ZodNumber") {
      columns.push(`${key} NUMERIC`);
    } else if (value._def.typeName === "ZodBoolean") {
      columns.push(`${key} INTEGER`);
    } else if (
      value._def.typeName === "ZodArray" ||
      value._def.typeName === "ZodObject"
    ) {
      columns.push(`${key}_json TEXT`);
    } else if (value._def.typeName === "ZodNullable") {
      const innerType = value.unwrap();
      if (
        innerType._def.typeName === "ZodObject" ||
        innerType._def.typeName === "ZodArray"
      ) {
        columns.push(`${key}_json TEXT NULL`);
      } else if (innerType._def.typeName === "ZodString") {
        columns.push(`${key} TEXT NULL`);
      } else if (innerType._def.typeName === "ZodNumber") {
        columns.push(`${key} NUMERIC NULL`);
      } else if (innerType._def.typeName === "ZodBoolean") {
        columns.push(`${key} INTEGER DEFAULT 0`);
      }
    }
  }

  return `CREATE TABLE IF NOT EXISTS ${tableName} (
    id TEXT PRIMARY KEY,
    ${columns.join(",")}
  )`;
}

export function objToXml(obj: any, parentKey?: string): string {
  const sanitizeTagName = (name: string): string => {
    return name.replace(/[^a-zA-Z0-9_]/g, "_").replace(
      /^[^a-zA-Z]/,
      "tag_$&",
    );
  };

  const wrapTag = (tag: string, content: string): string => {
    const safeTag = sanitizeTagName(tag);
    return `<${safeTag}>${content}</${safeTag}>`;
  };

  if (Array.isArray(obj)) {
    return wrapTag(
      `${parentKey ?? "array"}`,
      obj
        .map((item, index) =>
          wrapTag(
            "item",
            typeof item === "object"
              ? objToXml(item)
              : String(item),
          ),
        )
        .join(""),
    );
  }

  if (obj && typeof obj === "object") {
    if ("_def" in obj) {
      return objToSchemaInfo(obj as z.ZodType<any>, parentKey);
    }

    const entries = Object.entries(obj).filter(
      ([, v]) => v !== undefined && v !== null,
    );

    if (entries.length === 0) {
      return wrapTag(parentKey || "empty", "");
    }

    const it = entries
      .map(([key, value]) =>
        typeof value === "object"
          ? objToXml(value, key)
          : `<${key}>${value}</${key}>`,
      )
      .join("");

    return parentKey ? wrapTag(parentKey, it) : it;
  }

  return wrapTag(parentKey || "value", String(obj));
}

/**
 * Converts a Zod schema to XML schema info.
 * Corrects an issue with array handling by extracting
 * the inner type schema rather than re-passing the array,
 * which avoids infinite recursion and outputs the full schema.
 */
export function objToSchemaInfo(
  schema: z.ZodType<any>,
  parentKey?: string,
): string {
  const getFieldInfo = (field: z.ZodType<any>) => {
    // Get the base field by unwrapping optional/nullable
    let baseField = field;
    let isOptional = false;
    let isNullable = false;
    
    while (baseField._def.typeName === "ZodOptional" || baseField._def.typeName === "ZodNullable") {
      if (baseField._def.typeName === "ZodOptional") {
        isOptional = true;
      }
      if (baseField._def.typeName === "ZodNullable") {
        isNullable = true;
      }
      baseField = baseField.unwrap();
    }

    // Get description from the original field or unwrapped field
    const description = field.description || baseField.description || "";
    
    return { 
      description, 
      isRequired: !isOptional && !isNullable,
      isOptional,
      isNullable,
      baseField 
    };
  };

  const getTypeInfo = (field: z.ZodType<any>): string => {
    const baseField = field._def.typeName === "ZodOptional" || field._def.typeName === "ZodNullable" 
      ? field.unwrap() 
      : field;

    switch (baseField._def.typeName) {
      case "ZodString": return "string";
      case "ZodNumber": return "number";
      case "ZodBoolean": return "boolean";
      case "ZodArray": return "array";
      case "ZodObject": return "object";
      default: return "unknown";
    }
  };

  const buildFieldXml = (key: string = "", field: z.ZodType<any>): string => {
    const { description, isRequired, isOptional, isNullable, baseField } = getFieldInfo(field);
    const type = getTypeInfo(field);

    let content = "";
    if (description) content += `<description>${description}</description>`;
    content += dedent`
      <required>${String(isRequired)}</required>
      <type>${type}</type>
    `.replaceAll("\n", "");

    if (t(baseField) === "ZodObject") {
      content += `<fields>${buildObjectFields(baseField)}</fields>`;
    } else if (t(baseField) === "ZodArray") {
      let arrayField = baseField;
      while (
        arrayField._def.typeName === "ZodOptional" ||
        arrayField._def.typeName === "ZodNullable"
      ) {
        arrayField = arrayField.unwrap();
      }
      if (arrayField._def.typeName === "ZodArray") {
        const innerField = (arrayField as z.ZodArray<any>)._def.type;
        content += `<items>${buildFieldXml("item", innerField)}</items>`;
      }
    }

    return key ? `<${key}>${content}</${key}>` : content;
  };

  const buildObjectFields = (objSchema: z.ZodObject<any>): string => {
    return Object.entries(objSchema.shape)
      .map(([key, field]) => buildFieldXml(key, field))
      .join("");
  };

  switch (t(schema)) {
    case "ZodObject": {
      const fields = buildObjectFields(schema);
      return parentKey
        ? `<${parentKey}><fields>${fields}</fields></${parentKey}>`
        : `<schema><fields>${fields}</fields></schema>`;
    }
    case "ZodArray": {
      const items = buildFieldXml("item", schema);
      return parentKey ? `<${parentKey}>${items}</${parentKey}>` : items;
    }
    default: {
      return buildFieldXml(parentKey, schema);
    }
  }
}

function t(field: z.ZodType<any>) {
  if (
    field._def.typeName === "ZodOptional" ||
    field._def.typeName === "ZodNullable"
  ) {
    const unwrappedField = field.unwrap();
    return unwrappedField._def.typeName;
  }
  return field._def.typeName;
}

export function xmlToObj(element: string, schema: z.ZodObject<any>): any {
  const regex = new RegExp(
    `<${OUTPUT_DATA_WRAPPER}>(.*?)<\/${OUTPUT_DATA_WRAPPER}>`,
    "gs",
  );
  const matches = Array.from(element.matchAll(regex));
  const match = matches.pop();

  if (!match) {
    throw new Error(`No ${OUTPUT_DATA_WRAPPER} wrapper found in XML`);
  }

  function convertValue(value: string, fieldSchema?: z.ZodType<any>): any {
    if (!fieldSchema) return value;

    try {
      if (t(fieldSchema) === "ZodNumber") {
        const cleanNum = value.replace(/[^\d.-]/g, "");
        const num = Number(cleanNum);
        if (isNaN(num)) {
          throw new Error(`Invalid number: ${value}`);
        }
        return num;
      } else if (t(fieldSchema) === "ZodBoolean") {
        return value.toLowerCase() === "true";
      } else {
        return value.trim();
      }
    } catch (err) {
      console.error(`Type conversion failed for value: ${value}`, err);
      throw err;
    }
  }

  const parseNested = (
    content: string,
    currentSchema?: z.ZodType<any>,
  ): any => {
    content = content.trim();

    // Fix: If the schema is an array, use its inner type for conversion.
    if (
      currentSchema &&
      t(currentSchema) === "ZodArray" &&
      !content.startsWith("<array>")
    ) {
      if (content === "") return [];
      const itemMatches = Array.from(
        content.matchAll(/<item>(.*?)<\/item>/gs),
      );
      const innerSchema = currentSchema._def.type;
      if (itemMatches.length > 0) {
        return itemMatches.map((match) => {
          const itemContent = match[1].trim();
          return itemContent.includes("<")
            ? parseNested(itemContent, innerSchema)
            : convertValue(itemContent, innerSchema);
        });
      } else {
        return [convertValue(content, innerSchema)];
      }
    }

    if (content.startsWith("<array>")) {
      const arrayItems = content.match(/<item>(.*?)<\/item>/gs);
      if (!arrayItems) return [];
      const itemSchema =
        currentSchema && t(currentSchema) === "ZodArray"
          ? currentSchema._def.type
          : undefined;
      return arrayItems
        .map((item) => {
          const itemContent = item.match(/<item>(.*?)<\/item>/s)?.[1]?.trim();
          if (!itemContent) return null;
          return itemContent.includes("<")
            ? parseNested(itemContent, itemSchema)
            : convertValue(itemContent, itemSchema);
        })
        .filter((v) => v !== null);
    }

    if (
      currentSchema &&
      t(currentSchema) === "ZodArray" &&
      content === ""
    ) {
      return [];
    }

    const result: Record<string, any> = {};
    const fields = content.match(/<([^>/]+)>(.*?)<\/\1>/gs);

    if (fields) {
      fields.forEach((field) => {
        const m = field.match(/<([^>]+)>(.*?)<\/\1>/s);
        if (m) {
          const key = m[1];
          const value = m[2];
          let fieldSchema: z.ZodType<any> | undefined;
          if (currentSchema && t(currentSchema) === "ZodObject") {
            fieldSchema = currentSchema.shape[key];
          }
          if (fieldSchema) {
            if (
              t(fieldSchema) === "ZodArray" ||
              t(fieldSchema) === "ZodObject"
            ) {
              result[key] = parseNested(value, fieldSchema);
            } else {
              result[key] = convertValue(value, fieldSchema);
            }
          } else {
            if (
              value.includes("<array>") ||
              value.includes("<item>")
            ) {
              result[key] = parseNested(value, z.array(z.string()));
            } else if (value.match(/<[^>]+>/)) {
              result[key] = parseNested(value, z.object({}));
            } else {
              result[key] = value.trim();
            }
          }
        }
      });
      return result;
    }

    if (currentSchema) {
      return convertValue(content, currentSchema);
    }
    return content;
  };

  const parsed = parseNested(match[1], schema);
  return schema.parse(parsed);
}

/**
 * Validates and converts XML output against a schema
 * @param {string} xml - XML string to validate
 * @param {object} schema - Zod schema to validate against
 * @returns {object} Validation result with success/error info
 */
export function validateSchemaOutput(xml: string, schema: z.ZodObject<any>): {
  success: boolean;
  data?: any;
  error?: string;
} {
  try {
    const parsed = xmlToObj(xml, schema);
    const result = schema.parse(parsed);
    return {
      success: true,
      data: result,
    };
  } catch (err) {
    if (err instanceof z.ZodError) {
      const fieldErrors = err.issues
        .map((issue) =>
          `Field '${issue.path.join(".")}': ${issue.message}`,
        )
        .join("\n");
      return { success: false, error: fieldErrors };
    } else if (typeof err === "string") {
      try {
        const parsedErr = JSON.parse(err);
        if (Array.isArray(parsedErr)) {
          const fieldErrors = parsedErr
            .map((issue: any) =>
              `Field '${issue.path.join(".")}': ${issue.message}`,
            )
            .join("\n");
          return { success: false, error: fieldErrors };
        }
      } catch (_) {}
      return { success: false, error: err };
    }
    console.error(err);
    return { success: false, error: String(err) };
  }
}

export function normalizeZodType(zodType: z.ZodType<any>): any {
  const typeName = zodType._def.typeName;

  if (typeName === "ZodObject") {
    const objectSchema = zodType as z.ZodObject<any>;
    const shape = objectSchema.shape;
    const sortedKeys = Object.keys(shape).sort();
    const normalizedShape: Record<string, any> = {};
    for (const key of sortedKeys) {
      normalizedShape[key] = normalizeZodType(shape[key]);
    }
    return { type: "object", shape: normalizedShape };
  }

  if (typeName === "ZodArray") {
    return { type: "array", element: normalizeZodType(zodType._def.type) };
  }

  if (typeName === "ZodString") {
    return { type: "string" };
  }

  if (typeName === "ZodNumber") {
    return { type: "number" };
  }

  if (typeName === "ZodBoolean") {
    return { type: "boolean" };
  }

  if (typeName === "ZodOptional") {
    const inner = normalizeZodType(zodType.unwrap());
    return { ...inner, optional: true };
  }

  if (typeName === "ZodNullable") {
    const inner = normalizeZodType(zodType.unwrap());
    return { ...inner, nullable: true };
  }

  if (typeName === "ZodLiteral") {
    return { type: "literal", value: zodType._def.value };
  }

  if (typeName === "ZodEnum") {
    return { type: "enum", options: zodType._def.values };
  }

  if (typeName === "ZodUnion") {
    return { type: "union", options: zodType._def.options.map(normalizeZodType) };
  }

  return { type: typeName };
}

export function deepCompareSchemas(schema1: z.ZodObject<any>, schema2: z.ZodObject<any>): boolean {
  const normalized1 = normalizeZodType(schema1); // normalizeZodType(schema1);
  const normalized2 = normalizeZodType(schema2); // normalizeZodType(schema2);

  return JSON.stringify(normalized1) === JSON.stringify(normalized2);
}

export function validateSchemaMatch(
  name: string,
  fromSchema: z.ZodObject<any>,
  toSchema: z.ZodObject<any>,
  existingFromSchema: string,
  existingToSchema: string
) {
  const fromMatch = deepCompareSchemas(fromSchema, dezerialize(JSON.parse(existingFromSchema)));
  const toMatch = deepCompareSchemas(toSchema, dezerialize(JSON.parse(existingToSchema)));

  if (!fromMatch || !toMatch) {
    const differences = [];
    
    if (!fromMatch) {
      differences.push("input schema");
    }
    if (!toMatch) {
      differences.push("output schema");
    }

    throw new Error(
      `Schema mismatch for agent "${name}": The ${differences.join(" and ")} do not match the existing schemas.\n\n` +
      `To update schemas:\n` +
      `1. Delete the existing agent using Agent("${name}").delete()\n` +
      `2. Reinitialize with new schemas using Agent("${name}").init(newFromSchema, newToSchema)\n\n` +
      `This ensures data consistency and proper schema migration.`
    );
  }
}