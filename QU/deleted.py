    def prepare_results_folder(self, base_path, results_folder):
        '''Bereitet Pfad zum 'Results' Ordner vor und erstellt Ordner falls noch nicht vorhanden.'''
        if not os.path.exists(base_path):
            raise FileNotFoundError(f'Der angegebene Pfad existiert nicht: {base_path}')
        
        results_path = os.path.join(base_path, results_folder)
        os.makedirs(results_path, exist_ok=True)
        
        self.log(f'''Ordner '{results_folder}' ist vorhanden oder wurde erstellt unter: {results_path}''')
        return results_path