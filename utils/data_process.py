import yaml

def replace_path_in_yaml(input_file, output_file, old_prefix, new_prefix):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'data' in data:
            new_data = []
            for path in data['data']:
                if path.startswith(old_prefix):
                    new_path = path.replace(old_prefix, new_prefix, 1)
                    new_data.append(new_path)
                else:
                    new_data.append(path)
            data['data'] = new_data
        else:
            print("'data' key not found in YAML file.")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)
        
        print(f"Path replacement completed, new file saved as {output_file}")
    
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_yaml = './data/DomainNet/domainnet_train.yaml'  # input file path
    output_yaml = './data/DomainNet/domainnet_train_modified.yaml'  # output file path
    old_prefix = 'data/DomainNet/'
    new_prefix = './data/DomainNet/'
    
    replace_path_in_yaml(input_yaml, output_yaml, old_prefix, new_prefix)